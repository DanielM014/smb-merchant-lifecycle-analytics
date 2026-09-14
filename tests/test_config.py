from datetime import date
from pathlib import Path

import pytest

from smb_merchant_lifecycle.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


def test_loads_approved_project_config() -> None:
    config = load_config(CONFIG_PATH)

    assert config.project.seed == 3
    assert config.project.start_date == date(2025, 1, 1)
    assert config.project.end_date == date(2026, 8, 31)
    assert config.project.as_of_date == date(2026, 9, 1)
    assert config.project.merchant_count == 180
    assert config.project.currency == "COP"

    assert config.onboarding.first_transaction_window_days == 30

    assert config.sales_change.window_days == 28
    assert config.sales_change.minimum_history_days == 56
    assert config.sales_change.decline_threshold_pct == -20.0
    assert config.sales_change.growth_threshold_pct == 20.0
    assert config.sales_change.minimum_baseline_approved_transactions == 10
    assert config.sales_change.minimum_baseline_active_days == 7

    assert config.sales_change.priority.high_absolute_change_pct == 50.0
    assert config.sales_change.priority.medium_absolute_change_pct == 35.0
    assert config.sales_change.priority.low_absolute_change_pct == 20.0


def test_rejects_invalid_toml(tmp_path: Path) -> None:
    invalid_config_path = tmp_path / "invalid.toml"
    invalid_config_path.write_text(
        "[project]\nseed = 03\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(invalid_config_path)


def test_rejects_history_that_is_not_two_windows(
    tmp_path: Path,
) -> None:
    original_config = CONFIG_PATH.read_text(encoding="utf-8")
    expected_setting = "minimum_history_days = 56"

    assert expected_setting in original_config

    invalid_config = original_config.replace(
        expected_setting,
        "minimum_history_days = 55",
    )

    invalid_config_path = tmp_path / "invalid_history.toml"
    invalid_config_path.write_text(
        invalid_config,
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="minimum_history_days"):
        load_config(invalid_config_path)