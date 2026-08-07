from unittest.mock import patch

from innerthink.memory import EverOSClient


def test_remember_feedback_adds_then_flushes_one_session() -> None:
    client = EverOSClient()

    with patch.object(EverOSClient, "_post", side_effect=[{"ok": True}, {"ok": True}]) as post:
        result = client.remember_feedback("alice", "What is 2 + 2?", "4", "Correct")

    assert result["session_id"].startswith("feedback-")
    assert post.call_count == 2
    add_path, add_payload = post.call_args_list[0].args
    flush_path, flush_payload = post.call_args_list[1].args
    assert add_path == "/api/v2/memory/add"
    assert flush_path == "/api/v2/memory/flush"
    assert add_payload["messages"][0]["sender_id"] == "alice"
    assert flush_payload["session_id"] == add_payload["session_id"]
