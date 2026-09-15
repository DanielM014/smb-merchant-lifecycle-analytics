"""Modelo de ciclo de vida que integra onboarding y ventas."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.onboarding.modelo import (
    COLUMNAS_MODELO_ONBOARDING,
)
from smb_merchant_lifecycle.ventas.validacion import (
    validar_integridad_ventas,
)


ESTADOS_PRIMERA_TRANSACCION = frozenset(
    {
        "ON_TIME",
        "LATE",
        "NOT_OBSERVED",
        "NOT_APPLICABLE",
    }
)

COLUMNAS_MODELO_CICLO_VIDA = (
    *COLUMNAS_MODELO_ONBOARDING,
    "first_transaction_date",
    "days_to_first_transaction",
    "first_transaction_status",
    "commercial_activation_30d",
    "days_since_first_transaction",
    "is_active_30d_eligible",
    "is_active_30d",
)

_COLUMNAS_ENTERAS_ONBOARDING = (
    "onboarding_event_count",
    "days_since_registered",
    "days_in_current_stage",
    "days_to_technical_activation",
)


class ModeloCicloVidaError(ValueError):
    """Indica que las fuentes no permiten construir el ciclo de vida."""


def _texto(valor: object) -> str:
    """Convierte un valor escalar en texto sin representar nulos."""

    if pd.isna(valor):
        return ""

    return str(valor).strip()


def _booleano_estricto(valor: object, campo: str) -> bool:
    """Interpreta únicamente booleanos inequívocos."""

    if isinstance(valor, bool):
        return valor

    texto = _texto(valor).lower()

    if texto == "true":
        return True
    if texto == "false":
        return False

    raise ModeloCicloVidaError(
        f"{campo} debe contener únicamente True o False."
    )


def _fecha_timestamp(valor: object, campo: str) -> date:
    """Extrae la fecha calendario de un timestamp ISO validado."""

    try:
        return datetime.fromisoformat(
            _texto(valor)
        ).date()
    except ValueError as exc:
        raise ModeloCicloVidaError(
            f"{campo} debe contener timestamps ISO válidos."
        ) from exc


def _entero_nullable(
    serie: pd.Series,
    campo: str,
) -> pd.Series:
    """Normaliza una columna entera que puede contener valores vacíos."""

    normalizada = serie.map(_texto).replace("", pd.NA)

    try:
        return pd.to_numeric(
            normalizada,
            errors="raise",
        ).astype("Int64")
    except (TypeError, ValueError) as exc:
        raise ModeloCicloVidaError(
            f"{campo} debe contener enteros o valores vacíos."
        ) from exc


def _preparar_modelo_onboarding(
    modelo_onboarding: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Valida el grano y normaliza el modelo de onboarding."""

    if tuple(modelo_onboarding.columns) != (
        COLUMNAS_MODELO_ONBOARDING
    ):
        raise ModeloCicloVidaError(
            "merchant_onboarding_bi.csv no cumple el esquema esperado."
        )

    if modelo_onboarding.empty:
        raise ModeloCicloVidaError(
            "merchant_onboarding_bi.csv no puede estar vacío."
        )

    if modelo_onboarding["merchant_id"].duplicated().any():
        raise ModeloCicloVidaError(
            "merchant_id debe ser único en merchant_onboarding_bi.csv."
        )

    modelo = modelo_onboarding.copy(deep=True)
    modelo["is_technically_activated"] = modelo[
        "is_technically_activated"
    ].map(
        lambda valor: _booleano_estricto(
            valor,
            "is_technically_activated",
        )
    )
    modelo["is_commercial_activation_30d_eligible"] = modelo[
        "is_commercial_activation_30d_eligible"
    ].map(
        lambda valor: _booleano_estricto(
            valor,
            "is_commercial_activation_30d_eligible",
        )
    )

    for campo in _COLUMNAS_ENTERAS_ONBOARDING:
        modelo[campo] = _entero_nullable(
            modelo[campo],
            campo,
        )

    elegibilidad_esperada = modelo["registered_at"].map(
        lambda valor: (
            config.project.end_date
            - _fecha_timestamp(
                valor,
                "registered_at",
            )
        ).days
        >= config.onboarding.first_transaction_window_days
    )

    if not modelo[
        "is_commercial_activation_30d_eligible"
    ].equals(elegibilidad_esperada):
        raise ModeloCicloVidaError(
            "is_commercial_activation_30d_eligible no coincide con "
            "la última fecha observada; regenere el modelo de onboarding."
        )

    return modelo.sort_values(
        "merchant_id",
        kind="stable",
        ignore_index=True,
    )


def _validar_ventas_procesadas(
    ventas: pd.DataFrame,
    modelo_onboarding: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Exige que ventas vuelva a superar la puerta de calidad."""

    resultado = validar_integridad_ventas(
        ventas,
        modelo_onboarding,
        config,
    )

    if not resultado.violaciones.empty:
        reglas = ", ".join(
            sorted(
                set(
                    resultado.violaciones["rule_id"]
                )
            )
        )
        raise ModeloCicloVidaError(
            "No se puede construir el ciclo de vida porque ventas "
            f"contiene violaciones: {reglas}."
        )

    return resultado.ventas_validas


def _preparar_actividad(
    ventas_validas: pd.DataFrame,
) -> tuple[dict[str, date], dict[str, pd.DataFrame]]:
    """Deriva primeras transacciones y series diarias por comercio."""

    ventas = ventas_validas.copy(deep=True)
    ventas["__sales_date"] = ventas["sales_date"].map(
        date.fromisoformat
    )
    ventas["approved_transactions"] = pd.to_numeric(
        ventas["approved_transactions"],
        errors="raise",
    ).astype("int64")

    primeras = (
        ventas.loc[
            ventas["approved_transactions"].gt(0)
        ]
        .groupby("merchant_id")["__sales_date"]
        .min()
        .to_dict()
    )
    series = {
        str(merchant_id): grupo.reset_index(drop=True)
        for merchant_id, grupo in ventas.groupby(
            "merchant_id",
            sort=False,
        )
    }

    return primeras, series


def _estado_primera_transaccion(
    is_technically_activated: bool,
    days_to_first_transaction: int | None,
    window_days: int,
) -> str:
    """Clasifica la primera transacción sin mezclar elegibilidad."""

    if days_to_first_transaction is None:
        return (
            "NOT_OBSERVED"
            if is_technically_activated
            else "NOT_APPLICABLE"
        )

    if 0 <= days_to_first_transaction <= window_days:
        return "ON_TIME"

    return "LATE"


def _actividad_posterior_30d(
    serie: pd.DataFrame,
    first_transaction_date: date,
    window_days: int,
) -> bool:
    """Busca actividad entre los días 1 y 30 posteriores."""

    inicio = first_transaction_date + timedelta(days=1)
    fin = first_transaction_date + timedelta(
        days=window_days
    )
    ventana = serie["__sales_date"].between(
        inicio,
        fin,
        inclusive="both",
    )

    return bool(
        serie.loc[
            ventana,
            "approved_transactions",
        ].gt(0).any()
    )


def construir_modelo_ciclo_vida(
    modelo_onboarding: pd.DataFrame,
    ventas: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Integra onboarding y ventas en una fila por comercio."""

    modelo = _preparar_modelo_onboarding(
        modelo_onboarding,
        config,
    )
    ventas_validas = _validar_ventas_procesadas(
        ventas,
        modelo,
        config,
    )
    primeras, series = _preparar_actividad(
        ventas_validas
    )
    window_days = (
        config.onboarding.first_transaction_window_days
    )

    first_transaction_dates: list[str] = []
    days_to_first: list[int | None] = []
    first_transaction_statuses: list[str] = []
    commercial_activation: list[bool | None] = []
    days_since_first: list[int | None] = []
    active_eligible: list[bool] = []
    active_30d: list[bool | None] = []

    for _, fila in modelo.iterrows():
        merchant_id = str(fila["merchant_id"])
        registered_date = _fecha_timestamp(
            fila["registered_at"],
            "registered_at",
        )
        first_date = primeras.get(merchant_id)
        is_technically_activated = bool(
            fila["is_technically_activated"]
        )
        is_commercial_eligible = bool(
            fila[
                "is_commercial_activation_30d_eligible"
            ]
        )

        days_to = (
            (first_date - registered_date).days
            if first_date is not None
            else None
        )
        status = _estado_primera_transaccion(
            is_technically_activated,
            days_to,
            window_days,
        )
        is_active_eligible = bool(
            first_date is not None
            and (
                config.project.end_date - first_date
            ).days
            >= window_days
        )

        first_transaction_dates.append(
            first_date.isoformat()
            if first_date is not None
            else ""
        )
        days_to_first.append(days_to)
        first_transaction_statuses.append(status)
        commercial_activation.append(
            status == "ON_TIME"
            if is_commercial_eligible
            else None
        )
        days_since_first.append(
            (
                config.project.as_of_date
                - first_date
            ).days
            if first_date is not None
            else None
        )
        active_eligible.append(is_active_eligible)
        active_30d.append(
            _actividad_posterior_30d(
                series[merchant_id],
                first_date,
                window_days,
            )
            if is_active_eligible
            else None
        )

    modelo["first_transaction_date"] = (
        first_transaction_dates
    )
    modelo["days_to_first_transaction"] = pd.array(
        days_to_first,
        dtype="Int64",
    )
    modelo["first_transaction_status"] = (
        first_transaction_statuses
    )
    modelo["commercial_activation_30d"] = pd.array(
        commercial_activation,
        dtype="boolean",
    )
    modelo["days_since_first_transaction"] = pd.array(
        days_since_first,
        dtype="Int64",
    )
    modelo["is_active_30d_eligible"] = active_eligible
    modelo["is_active_30d"] = pd.array(
        active_30d,
        dtype="boolean",
    )

    if not set(
        modelo["first_transaction_status"]
    ).issubset(ESTADOS_PRIMERA_TRANSACCION):
        raise AssertionError(
            "El modelo produjo estados de primera transacción inválidos."
        )

    if not modelo["merchant_id"].is_unique:
        raise AssertionError(
            "merchant_lifecycle_bi.csv contiene merchant_id duplicados."
        )

    return modelo.loc[
        :,
        list(COLUMNAS_MODELO_CICLO_VIDA),
    ]


def guardar_modelo_ciclo_vida(
    modelo: pd.DataFrame,
    ruta: str | Path = Path(
        "data/procesada/merchant_lifecycle_bi.csv"
    ),
) -> Path:
    """Guarda el modelo integrado de forma reproducible."""

    if tuple(modelo.columns) != COLUMNAS_MODELO_CICLO_VIDA:
        raise ModeloCicloVidaError(
            "El modelo de ciclo de vida no cumple el esquema esperado."
        )

    ruta_salida = Path(ruta)
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    modelo.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return ruta_salida


def main() -> None:
    """Construye el ciclo de vida desde las salidas procesadas."""

    config = load_config()
    modelo_onboarding = pd.read_csv(
        "data/procesada/merchant_onboarding_bi.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    ventas = pd.read_csv(
        "data/procesada/merchant_sales_daily.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    modelo = construir_modelo_ciclo_vida(
        modelo_onboarding,
        ventas,
        config,
    )
    ruta = guardar_modelo_ciclo_vida(modelo)
    con_primera_transaccion = int(
        modelo["first_transaction_date"].ne("").sum()
    )

    print(
        "Modelo de ciclo de vida generado: "
        f"{len(modelo)} comercios, "
        f"{con_primera_transaccion} con primera transacción "
        f"en {ruta}."
    )


if __name__ == "__main__":
    main()
