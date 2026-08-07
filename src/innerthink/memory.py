import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EverOSClientError(RuntimeError):
    """Raised when the configured EverOS service cannot satisfy a request."""


@dataclass(frozen=True)
class EverOSClient:
    base_url: str = "http://127.0.0.1:8001"
    timeout_seconds: float = 120.0
    app_id: str = "innerthink"
    project_id: str = "default"

    @classmethod
    def from_env(cls) -> "EverOSClient":
        return cls(
            base_url=os.getenv(
                "INNERTHINK_EVEROS_URL",
                "http://127.0.0.1:8001",
            ).rstrip("/"),
            timeout_seconds=float(os.getenv("INNERTHINK_EVEROS_TIMEOUT_SECONDS", "120")),
            app_id=os.getenv("INNERTHINK_EVEROS_APP_ID", "innerthink"),
            project_id=os.getenv("INNERTHINK_EVEROS_PROJECT_ID", "default"),
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {"status": "ok"}
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise EverOSClientError(f"EverOS returned HTTP {error.code}: {body}") from error
        except URLError as error:
            raise EverOSClientError(
                f"Cannot reach EverOS at {self.base_url}. Start scripts/start-everos.sh: "
                f"{error.reason}"
            ) from error

    def recall(self, user_id: str, query: str, *, top_k: int = 5) -> dict[str, Any]:
        return self._post(
            "/api/v2/memory/search",
            {
                "user_id": user_id,
                "app_id": self.app_id,
                "project_id": self.project_id,
                "query": query,
                "top_k": top_k,
            },
        )

    def remember_feedback(
        self,
        user_id: str,
        prompt: str,
        answer: str,
        feedback: str,
    ) -> dict[str, Any]:
        session_id = f"feedback-{uuid.uuid4()}"
        content = (
            "Reasoning feedback. "
            f"Task: {prompt}\nAnswer: {answer}\nUser feedback: {feedback}"
        )
        added = self._post(
            "/api/v2/memory/add",
            {
                "session_id": session_id,
                "app_id": self.app_id,
                "project_id": self.project_id,
                "messages": [
                    {
                        "sender_id": user_id,
                        "role": "user",
                        "timestamp": int(time.time() * 1000),
                        "content": content,
                    }
                ],
            },
        )
        flushed = self._post(
            "/api/v2/memory/flush",
            {
                "session_id": session_id,
                "app_id": self.app_id,
                "project_id": self.project_id,
            },
        )
        return {"session_id": session_id, "added": added, "flushed": flushed}
