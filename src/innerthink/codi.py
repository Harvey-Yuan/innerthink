from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from innerthink.interventions import IdentityHook, LatentHook

GenerationMode = Literal["direct", "latent", "verbalized"]


@dataclass
class GenerationOutput:
    sequences: torch.LongTensor
    latent_vectors: list[torch.Tensor]


class CodiQwen(nn.Module):
    """Minimal CODI architecture matching the published Qwen3-8B checkpoint."""

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]

    def __init__(
        self,
        base_model_id: str,
        checkpoint_id: str,
        *,
        dtype: torch.dtype,
        cache_dir: str | None = None,
        token: str | None = None,
        lora_r: int = 128,
        lora_alpha: int = 32,
        projection_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_id,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=True,
        )
        self._configure_tokenizer(self.tokenizer)

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            cache_dir=cache_dir,
            token=token,
            low_cpu_mem_usage=True,
        )
        base.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=self.target_modules,
            init_lora_weights=True,
        )
        # Keep this attribute name identical to the training wrapper. The released
        # state dict is keyed under "codi.*".
        self.codi = get_peft_model(base, lora_config)
        hidden_size = self.codi.config.hidden_size
        if hidden_size != projection_dim:
            raise ValueError(
                f"Checkpoint expects hidden/projection size {projection_dim}, "
                f"but {base_model_id} reports {hidden_size}."
            )
        self.prj = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, hidden_size),
        )
        # The training implementation adds this with an explicit name, so the
        # checkpoint keys are prj.ln.{weight,bias}, not prj.4.*.
        self.prj.add_module("ln", nn.LayerNorm(hidden_size))
        self.prj.to(dtype=dtype)

    @staticmethod
    def _configure_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        required = ("<|bocot|>", "<|eocot|>")
        missing = [
            token
            for token in required
            if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id
        ]
        if missing:
            raise ValueError(f"CODI checkpoint tokenizer is missing special tokens: {missing}")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _project(self, latent: torch.Tensor) -> torch.Tensor:
        # LayerNorm may promote on some backends. The transformer expects its own dtype.
        return self.prj(latent).to(dtype=self.codi.dtype)

    def _delimiter_ids(self, mode: GenerationMode) -> list[int]:
        end_id = self.tokenizer.convert_tokens_to_ids("<|eocot|>")
        if mode == "direct":
            return [self.tokenizer.eos_token_id, end_id]
        start_id = self.tokenizer.convert_tokens_to_ids("<|bocot|>")
        return [self.tokenizer.eos_token_id, start_id]

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        *,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.LongTensor:
        if greedy:
            return torch.argmax(logits, dim=-1)
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero when sampling")

        logits = logits / temperature
        if top_k > 0:
            keep = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, keep, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            remove_original = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
            logits = logits.masked_fill(remove_original, -torch.inf)

        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        mode: GenerationMode,
        max_new_tokens: int,
        latent_iterations: int = 6,
        greedy: bool = True,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.95,
        latent_hook: LatentHook | None = None,
        return_latent_vectors: bool = False,
    ) -> GenerationOutput:
        if mode not in {"direct", "latent", "verbalized"}:
            raise ValueError(f"Unsupported generation mode: {mode}")

        delimiter = torch.tensor(
            self._delimiter_ids(mode),
            dtype=torch.long,
            device=input_ids.device,
        ).expand(input_ids.shape[0], -1)
        prefixed_ids = torch.cat((input_ids, delimiter), dim=1)
        prefixed_mask = torch.cat((attention_mask, torch.ones_like(delimiter)), dim=1)

        if mode in {"direct", "verbalized"}:
            generation_args: dict[str, object] = {
                "input_ids": prefixed_ids,
                "attention_mask": prefixed_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": not greedy,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if not greedy:
                generation_args.update(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            else:
                # Qwen's bundled generation config carries sampling defaults.
                # Explicitly clear them to avoid misleading "ignored" warnings.
                generation_args.update(temperature=None, top_k=None, top_p=None)
            outputs = self.codi.generate(**generation_args)
            return GenerationOutput(
                sequences=outputs[:, prefixed_ids.shape[1] :],
                latent_vectors=[],
            )

        hook = latent_hook or IdentityHook()
        outputs = self.codi(
            input_ids=prefixed_ids,
            attention_mask=prefixed_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        latent = self._project(outputs.hidden_states[-1][:, -1:, :])
        latent = hook(0, latent)
        latent_vectors = [latent.detach().clone()] if return_latent_vectors else []

        # This reproduces the public implementation: the initial projected state is
        # followed by `latent_iterations` recurrent passes.
        for step in range(1, latent_iterations + 1):
            outputs = self.codi(
                inputs_embeds=latent,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent = hook(step, self._project(outputs.hidden_states[-1][:, -1:, :]))
            if return_latent_vectors:
                latent_vectors.append(latent.detach().clone())

        end_id = self.tokenizer.convert_tokens_to_ids("<|eocot|>")
        end_ids = torch.full(
            (input_ids.shape[0],),
            end_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        next_embedding = self.codi.get_input_embeddings()(end_ids).unsqueeze(1)

        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        generated: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
        for _ in range(max_new_tokens):
            outputs = self.codi(
                inputs_embeds=next_embedding,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_ids = self._sample(
                logits,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            for batch_index, token_id in enumerate(next_ids.tolist()):
                if not finished[batch_index]:
                    generated[batch_index].append(token_id)
                    if token_id == self.tokenizer.eos_token_id:
                        finished[batch_index] = True
            if bool(finished.all().item()):
                break
            next_embedding = self.codi.get_input_embeddings()(next_ids).unsqueeze(1)

        max_length = max((len(tokens) for tokens in generated), default=0)
        sequences = torch.full(
            (input_ids.shape[0], max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        for batch_index, tokens in enumerate(generated):
            if tokens:
                sequences[batch_index, : len(tokens)] = torch.tensor(
                    tokens,
                    dtype=torch.long,
                    device=input_ids.device,
                )
        return GenerationOutput(sequences=sequences, latent_vectors=latent_vectors)
