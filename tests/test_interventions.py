import torch

from innerthink.interventions import RecordingHook, ScaleStepHook


def test_recording_hook_records_and_transforms_selected_step() -> None:
    hook = RecordingHook(ScaleStepHook(step=1, scale=0.0))

    first = hook(0, torch.tensor([[[3.0, 4.0]]]))
    second = hook(1, torch.tensor([[[1.0, 2.0]]]))

    assert torch.equal(first, torch.tensor([[[3.0, 4.0]]]))
    assert torch.equal(second, torch.zeros(1, 1, 2))
    assert hook.steps[0].l2_norm == 5.0
    assert hook.steps[0].cosine_from_previous is None
    assert hook.steps[1].l2_norm == 0.0
    assert hook.steps[1].cosine_from_previous == 0.0
