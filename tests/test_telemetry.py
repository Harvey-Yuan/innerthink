from innerthink.telemetry import SnowflakeTelemetry


def test_snowflake_telemetry_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setenv("INNERTHINK_SNOWFLAKE_ENABLED", "false")
    assert SnowflakeTelemetry.from_env() is None


def test_snowflake_telemetry_requires_complete_configuration(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INNERTHINK_SNOWFLAKE_ENABLED", "true")
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)

    try:
        SnowflakeTelemetry.from_env()
    except ValueError as error:
        assert "SNOWFLAKE_ACCOUNT" in str(error)
    else:
        raise AssertionError("Expected incomplete Snowflake configuration to fail")


def test_snowflake_telemetry_loads_dotenv_and_normalizes_account(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "INNERTHINK_SNOWFLAKE_ENABLED=true",
                "SNOWFLAKE_ACCOUNT=organization/account",
                "SNOWFLAKE_USER=builder",
                "SNOWFLAKE_PASSWORD=secret",
                "SNOWFLAKE_WAREHOUSE=DEFAULT$",
                "SNOWFLAKE_DATABASE=USER$",
                "SNOWFLAKE_SCHEMA=PUBLIC",
                "SNOWFLAKE_ROLE=PUBLIC",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    telemetry = SnowflakeTelemetry.from_env()

    assert telemetry is not None
    assert telemetry.connection_options["account"] == "organization-account"
    assert telemetry.connection_options["role"] == "PUBLIC"
