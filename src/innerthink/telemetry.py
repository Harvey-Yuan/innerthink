import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from innerthink.runtime import InferenceResult

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_READY_TABLES: set[tuple[str, str, str, str, str]] = set()
_READY_TABLES_LOCK = Lock()


class _SnowflakeSettings(BaseSettings):
    """Snowflake settings shared by the local API, CLI, and Cursor MCP server."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    enabled: bool = Field(default=False, validation_alias="INNERTHINK_SNOWFLAKE_ENABLED")
    table: str = Field(default="INFERENCE_EVENTS", validation_alias="INNERTHINK_SNOWFLAKE_TABLE")
    account: str | None = Field(default=None, validation_alias="SNOWFLAKE_ACCOUNT")
    user: str | None = Field(default=None, validation_alias="SNOWFLAKE_USER")
    password: str | None = Field(default=None, validation_alias="SNOWFLAKE_PASSWORD")
    warehouse: str | None = Field(default=None, validation_alias="SNOWFLAKE_WAREHOUSE")
    database: str | None = Field(default=None, validation_alias="SNOWFLAKE_DATABASE")
    schema_name: str | None = Field(default=None, validation_alias="SNOWFLAKE_SCHEMA")
    role: str | None = Field(default=None, validation_alias="SNOWFLAKE_ROLE")


@dataclass(frozen=True)
class SnowflakeTelemetry:
    connection_options: dict[str, str]
    table: str = "INFERENCE_EVENTS"

    @classmethod
    def from_env(cls) -> "SnowflakeTelemetry | None":
        settings = _SnowflakeSettings()
        if not settings.enabled:
            return None
        mapping = {
            "account": ("SNOWFLAKE_ACCOUNT", settings.account),
            "user": ("SNOWFLAKE_USER", settings.user),
            "password": ("SNOWFLAKE_PASSWORD", settings.password),
            "warehouse": ("SNOWFLAKE_WAREHOUSE", settings.warehouse),
            "database": ("SNOWFLAKE_DATABASE", settings.database),
            "schema": ("SNOWFLAKE_SCHEMA", settings.schema_name),
        }
        options = {key: value or "" for key, (_, value) in mapping.items()}
        missing = [env_name for env_name, value in mapping.values() if not value]
        if missing:
            raise ValueError(f"Missing Snowflake settings: {', '.join(missing)}")
        # Snowsight URLs often display organization/account as "org/account".
        # The Python connector expects the same identifier with a hyphen.
        options["account"] = options["account"].replace("/", "-")
        if settings.role:
            options["role"] = settings.role
        table = settings.table
        if not _IDENTIFIER.fullmatch(table):
            raise ValueError("INNERTHINK_SNOWFLAKE_TABLE must be a simple SQL identifier")
        return cls(connection_options=options, table=table)

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @property
    def _table_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.connection_options["account"],
            self.connection_options["database"],
            self.connection_options["schema"],
            self.connection_options.get("role", ""),
            self.table,
        )

    def _ensure_table(self, cursor: Any) -> None:
        key = self._table_key
        with _READY_TABLES_LOCK:
            if key in _READY_TABLES:
                return
        cursor.execute(
            f"""
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
        )
        with _READY_TABLES_LOCK:
            _READY_TABLES.add(key)

    def check_connection(self) -> dict[str, str | None]:
        """Verify credentials and prepare the telemetry table without model inference."""
        import snowflake.connector

        query = """
            SELECT
                CURRENT_ACCOUNT() AS ACCOUNT,
                CURRENT_USER() AS USER_NAME,
                CURRENT_ROLE() AS ROLE,
                CURRENT_WAREHOUSE() AS WAREHOUSE,
                CURRENT_DATABASE() AS DATABASE_NAME,
                CURRENT_SCHEMA() AS SCHEMA_NAME
        """
        with (
            snowflake.connector.connect(**self.connection_options) as connection,
            connection.cursor() as cursor,
        ):
            self._ensure_table(cursor)
            cursor.execute(query)
            row = cursor.fetchone()
            columns = [column[0].lower() for column in cursor.description]
        return {"table": self.table, **dict(zip(columns, row, strict=True))}

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
            self._ensure_table(cursor)
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
            self._ensure_table(cursor)
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
