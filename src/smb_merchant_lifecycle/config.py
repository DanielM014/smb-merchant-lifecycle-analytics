"""Validación de la configuración del proyecto."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the project configuration is missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """General settings for synthetic data generation."""

    seed: int
    start_date: date
    end_date: date
    as_of_date: date
    merchant_count: int
    currency: str


@dataclass(frozen=True, slots=True)
class OnboardingSettings:
    """Rules for measuring onboarding success."""

    first_transaction_window_days: int


@dataclass(frozen=True, slots=True)
class SalesChangePrioritySettings:
    """Absolute percentage thresholds used to prioritize directional insights."""

    high_absolute_change_pct: float
    medium_absolute_change_pct: float
    low_absolute_change_pct: float


@dataclass(frozen=True, slots=True)
class SalesChangeSettings:
    """Rules for comparing consecutive merchant sales windows."""

    window_days: int
    minimum_history_days: int
    decline_threshold_pct: float
    growth_threshold_pct: float
    minimum_baseline_approved_transactions: int
    minimum_baseline_active_days: int
    priority: SalesChangePrioritySettings


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Complete validated project configuration."""

    project: ProjectSettings
    onboarding: OnboardingSettings
    sales_change: SalesChangeSettings


def _require_table(
    container: Mapping[str, Any],
    key: str,
    parent: str,
) -> Mapping[str, Any]:
    value = container.get(key)

    if not isinstance(value, dict):
        raise ConfigError(f"{parent}.{key} must be a TOML table.")

    return value


def _require_int(
    table: Mapping[str, Any],
    key: str,
    table_name: str,
) -> int:
    if key not in table:
        raise ConfigError(f"Missing required setting: {table_name}.{key}.")

    value = table[key]

    # bool is a subclass of int in Python, so an exact type check is required.
    if type(value) is not int:
        raise ConfigError(f"{table_name}.{key} must be an integer.")

    return value


def _require_number(
    table: Mapping[str, Any],
    key: str,
    table_name: str,
) -> float:
    if key not in table:
        raise ConfigError(f"Missing required setting: {table_name}.{key}.")

    value = table[key]

    if type(value) not in (int, float):
        raise ConfigError(f"{table_name}.{key} must be numeric.")

    return float(value)


def _require_string(
    table: Mapping[str, Any],
    key: str,
    table_name: str,
) -> str:
    if key not in table:
        raise ConfigError(f"Missing required setting: {table_name}.{key}.")

    value = table[key]

    if not isinstance(value, str):
        raise ConfigError(f"{table_name}.{key} must be a string.")

    return value


def _require_date(
    table: Mapping[str, Any],
    key: str,
    table_name: str,
) -> date:
    value = _require_string(table, key, table_name)

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(
            f"{table_name}.{key} must use ISO format YYYY-MM-DD."
        ) from exc


def _validate_config(config: ProjectConfig) -> ProjectConfig:
    project = config.project
    onboarding = config.onboarding
    sales_change = config.sales_change
    priority = sales_change.priority

    if project.seed < 0:
        raise ConfigError("project.seed must be greater than or equal to zero.")

    if project.start_date > project.end_date:
        raise ConfigError(
            "project.start_date cannot be later than project.end_date."
        )

    if project.end_date > project.as_of_date:
        raise ConfigError(
            "project.end_date cannot be later than project.as_of_date."
        )

    if project.merchant_count <= 0:
        raise ConfigError("project.merchant_count must be greater than zero.")

    if (
        len(project.currency) != 3
        or not project.currency.isalpha()
        or not project.currency.isupper()
    ):
        raise ConfigError(
            "project.currency must be a three-letter uppercase currency code."
        )

    if onboarding.first_transaction_window_days <= 0:
        raise ConfigError(
            "onboarding.first_transaction_window_days must be greater than zero."
        )

    if sales_change.window_days <= 0:
        raise ConfigError(
            "sales_change.window_days must be greater than zero."
        )

    expected_history_days = 2 * sales_change.window_days

    if sales_change.minimum_history_days != expected_history_days:
        raise ConfigError(
            "sales_change.minimum_history_days must equal "
            "2 * sales_change.window_days "
            f"({expected_history_days})."
        )

    if sales_change.minimum_baseline_approved_transactions <= 0:
        raise ConfigError(
            "sales_change.minimum_baseline_approved_transactions "
            "must be greater than zero."
        )

    if not (
        1
        <= sales_change.minimum_baseline_active_days
        <= sales_change.window_days
    ):
        raise ConfigError(
            "sales_change.minimum_baseline_active_days must be between "
            "1 and sales_change.window_days."
        )

    percentage_settings = (
        sales_change.decline_threshold_pct,
        sales_change.growth_threshold_pct,
        priority.low_absolute_change_pct,
        priority.medium_absolute_change_pct,
        priority.high_absolute_change_pct,
    )

    if not all(isfinite(value) for value in percentage_settings):
        raise ConfigError("Percentage thresholds must be finite numbers.")

    if sales_change.decline_threshold_pct >= 0:
        raise ConfigError(
            "sales_change.decline_threshold_pct must be negative."
        )

    if sales_change.growth_threshold_pct <= 0:
        raise ConfigError(
            "sales_change.growth_threshold_pct must be positive."
        )

    if not (
        0
        < priority.low_absolute_change_pct
        < priority.medium_absolute_change_pct
        < priority.high_absolute_change_pct
    ):
        raise ConfigError(
            "Priority thresholds must satisfy "
            "0 < low < medium < high."
        )

    minimum_signal_threshold = min(
        abs(sales_change.decline_threshold_pct),
        abs(sales_change.growth_threshold_pct),
    )

    if priority.low_absolute_change_pct > minimum_signal_threshold:
        raise ConfigError(
            "The LOW priority threshold cannot be greater than the smallest "
            "directional insight threshold."
        )

    return config


def load_config(
    path: str | Path = Path("config/project.toml"),
) -> ProjectConfig:
    """Load and validate the project TOML configuration."""

    config_path = Path(path)

    try:
        with config_path.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"Configuration file not found: {config_path}."
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Invalid TOML in {config_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Could not read configuration file {config_path}: {exc}"
        ) from exc

    project_table = _require_table(raw_config, "project", "root")
    onboarding_table = _require_table(raw_config, "onboarding", "root")
    sales_change_table = _require_table(raw_config, "sales_change", "root")
    priority_table = _require_table(
        sales_change_table,
        "priority",
        "sales_change",
    )

    config = ProjectConfig(
        project=ProjectSettings(
            seed=_require_int(project_table, "seed", "project"),
            start_date=_require_date(
                project_table,
                "start_date",
                "project",
            ),
            end_date=_require_date(
                project_table,
                "end_date",
                "project",
            ),
            as_of_date=_require_date(
                project_table,
                "as_of_date",
                "project",
            ),
            merchant_count=_require_int(
                project_table,
                "merchant_count",
                "project",
            ),
            currency=_require_string(
                project_table,
                "currency",
                "project",
            ),
        ),
        onboarding=OnboardingSettings(
            first_transaction_window_days=_require_int(
                onboarding_table,
                "first_transaction_window_days",
                "onboarding",
            ),
        ),
        sales_change=SalesChangeSettings(
            window_days=_require_int(
                sales_change_table,
                "window_days",
                "sales_change",
            ),
            minimum_history_days=_require_int(
                sales_change_table,
                "minimum_history_days",
                "sales_change",
            ),
            decline_threshold_pct=_require_number(
                sales_change_table,
                "decline_threshold_pct",
                "sales_change",
            ),
            growth_threshold_pct=_require_number(
                sales_change_table,
                "growth_threshold_pct",
                "sales_change",
            ),
            minimum_baseline_approved_transactions=_require_int(
                sales_change_table,
                "minimum_baseline_approved_transactions",
                "sales_change",
            ),
            minimum_baseline_active_days=_require_int(
                sales_change_table,
                "minimum_baseline_active_days",
                "sales_change",
            ),
            priority=SalesChangePrioritySettings(
                high_absolute_change_pct=_require_number(
                    priority_table,
                    "high_absolute_change_pct",
                    "sales_change.priority",
                ),
                medium_absolute_change_pct=_require_number(
                    priority_table,
                    "medium_absolute_change_pct",
                    "sales_change.priority",
                ),
                low_absolute_change_pct=_require_number(
                    priority_table,
                    "low_absolute_change_pct",
                    "sales_change.priority",
                ),
            ),
        ),
    )

    return _validate_config(config)