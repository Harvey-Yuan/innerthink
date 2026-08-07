import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from innerthink.runtime import InferenceResult

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


@dataclass(frozen=True)
class SnowflakeTelemetry:
    connection_options: dict[str, str]
    table: str = "INFERENCE_EVENTS"

    @classmethod
    def from_env(cls) -> "SnowflakeTelemetry | None":
        enabled = os.getenv("INNERTHINK_SNOWFLAKE_ENABLED", "false").lower()
        if enabled not in {"1", "true", "yes"}:
            return None
        mapping = {
            "account": "SNOWFLAKE_ACCOUNT",
            "user": "SNOWFLAKE_USER",
            "password": "SNOWFLAKE_PASSWORD",
            "warehouse": "SNOWFLAKE_WAREHOUSE",
            "database": "SNOWFLAKE_DATABASE",
            "schema": "SNOWFLAKE_SCHEMA",
        }
        options = {key: os.getenv(env_name, "") for key, env_name in mapping.items()}
        missing = [mapping[key] for key, value in options.items() if not value]
        if missing:
            raise ValueError(f"Missing Snowflake settings: {', '.join(missing)}")
        role = os.getenv("SNOWFLAKE_ROLE")
        if role:
            options["role"] = role
        table = os.getenv("INNERTHINK_SNOWFLAKE_TABLE", "INFERENCE_EVENTS")
        if not _IDENTIFIER.fullmatch(table):
            raise ValueError("INNERTHINK_SNOWFLAKE_TABLE must be a simple SQL identifier")
        return cls(connection_options=options, table=table)

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def record(
        self,
        result: "InferenceResult",
        *,
        prompt: str,
        model_id: str,
        intervention_step: int | None = None,
        intervention_scale: float | None = None,
    ) -> None:
        import snowflake.connector

        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                EVENT_ID STRING,
                OCCURRED_AT TIMESTAMP_TZ,
                PROMPT_HASH STRING,
                ANSWER_HASH STRING,
                MODEL_ID STRING,
                MODE STRING,
                ELAPSED_MS FLOAT,
                PROMPT_TOKENS NUMBER,
                OUTPUT_TOKENS NUMBER,
                VISIBLE_REASONING_TOKENS NUMBER,
                LATENT_ITERATIONS NUMBER,
                LATENT_STATES NUMBER,
                INTERVENTION_STEP NUMBER,
                INTERVENTION_SCALE FLOAT
            )
        """
        insert_sql = f"""
            INSERT INTO {self.table} (
                EVENT_ID, OCCURRED_AT, PROMPT_HASH, ANSWER_HASH, MODEL_ID, MODE,
                ELAPSED_MS, PROMPT_TOKENS, OUTPUT_TOKENS, VISIBLE_REASONING_TOKENS,
                LATENT_ITERATIONS, LATENT_STATES, INTERVENTION_STEP, INTERVENTION_SCALE
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            str(uuid.uuid4()),
            datetime.now(UTC),
            self._hash(prompt),
            self._hash(result.answer),
            model_id,
            result.mode,
            result.elapsed_ms,
            result.prompt_tokens,
            result.output_tokens,
            result.visible_reasoning_tokens,
            result.latent_iterations,
            result.latent_states,
            intervention_step,
            intervention_scale,
        )
        with (
            snowflake.connector.connect(**self.connection_options) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(create_sql)
            cursor.execute(insert_sql, values)

    def summary(self) -> dict[str, Any]:
        import snowflake.connector

        query = f"""
            SELECT
                MODEL_ID,
                MODE,
                COUNT(*) AS RUNS,
                ROUND(AVG(ELAPSED_MS), 2) AS AVG_ELAPSED_MS,
                SUM(PROMPT_TOKENS) AS PROMPT_TOKENS,
                SUM(OUTPUT_TOKENS) AS OUTPUT_TOKENS,
                SUM(LATENT_ITERATIONS) AS LATENT_ITERATIONS,
                COUNT_IF(INTERVENTION_STEP IS NOT NULL) AS INTERVENTION_RUNS
            FROM {self.table}
            GROUP BY MODEL_ID, MODE
            ORDER BY RUNS DESC
        """
        with (
            snowflake.connector.connect(**self.connection_options) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(query)
            columns = [column[0].lower() for column in cursor.description]
            rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        return {"table": self.table, "groups": rows}


def record_safely(
    telemetry: SnowflakeTelemetry,
    result: "InferenceResult",
    **kwargs: Any,
) -> None:
    try:
        telemetry.record(result, **kwargs)
    except Exception:
        logger.exception("Snowflake telemetry failed; inference result remains available")
