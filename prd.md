

https://huggingface.co/cds-jb/codi_qwen3-8b-answer_only



PRD: CODI Inner Thinking Demo
Goal
Build an engaging hackathon demo showing how CODI-Qwen3-8B reasons through hidden states while producing fewer visible reasoning tokens.
Demo
Users select a reasoning problem and compare:
* Qwen3-8B with explicit reasoning
* CODI-Qwen3-8B with latent reasoning
The UI displays:
* Final answer and correctness
* Visible reasoning tokens
* Token reduction percentage
* Inference latency
* Estimated cost
Datasets
* GSM8K
* MATH-500
* BBH subset
Key Experience
Visualize CODI’s hidden reasoning as animated latent steps rather than fabricated text. End each comparison with a clear result:
Same answer quality, 62% fewer visible tokens.
Success Criteria
* Run CODI-Qwen3-8B locally
* Support at least three datasets
* Show side-by-side results within seconds
* Demonstrate measurable token reduction without significant accuracy loss
* Deliver a memorable five-minute live demo
