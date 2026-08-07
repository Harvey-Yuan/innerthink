from unittest.mock import patch

from innerthink.client import InnerThinkClient


def test_intervene_compares_baseline_and_changed_answers() -> None:
    baseline = {"answer": "4", "elapsed_ms": 10.0}
    changed = {"answer": "5", "elapsed_ms": 12.5}
    client = InnerThinkClient()

    with patch.object(InnerThinkClient, "generate", side_effect=[baseline, changed]) as generate:
        result = client.intervene("What is 2 + 2?", step=3, scale=0.0)

    assert result["answer_changed"] is True
    assert result["elapsed_delta_ms"] == 2.5
    assert generate.call_count == 2
    assert generate.call_args_list[1].kwargs["intervention_step"] == 3
    assert generate.call_args_list[1].kwargs["intervention_scale"] == 0.0
