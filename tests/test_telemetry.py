from innerthink.telemetry import SnowflakeTelemetry


def test_snowflake_telemetry_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("INNERTHINK_SNOWFLAKE_ENABLED", raising=False)
    assert SnowflakeTelemetry.from_env() is None


def test_snowflake_telemetry_requires_complete_configuration(monkeypatch) -> None:
    monkeypatch.setenv("INNERTHINK_SNOWFLAKE_ENABLED", "true")
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)

    try:
        SnowflakeTelemetry.from_env()
    except ValueError as error:
        assert "SNOWFLAKE_ACCOUNT" in str(error)
    else:
        raise AssertionError("Expected incomplete Snowflake configuration to fail")
