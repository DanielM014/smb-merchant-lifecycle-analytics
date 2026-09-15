"""Modelo analítico de onboarding con una fila por comercio."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from smb_merchant_lifecycle.config import ProjectConfig, load_config
from smb_merchant_lifecycle.onboarding.validacion import (
    validar_integridad_onboarding,
)


ETAPAS_PRINCIPALES = (
    "REGISTERED",
    "VALIDATION_STARTED",
    "DOCUMENTS_COMPLETED",
    "APPROVED",
    "ACTIVATED",
)

ORDEN_ETAPAS = {
    etapa: posicion
    for posicion, etapa in enumerate(
        ETAPAS_PRINCIPALES,
        start=1,
    )
}

EVENTOS_DE_SALIDA = frozenset(
    {
        "REJECTED",
        "ABANDONED",
    }
)

COLUMNAS_MODELO_ONBOARDING = (
    "merchant_id",
    "merchant_name",
    "city",
    "business_segment",
    "acquisition_channel",
    "registered_cohort_month",
    "registered_at",
    "validation_started_at",
    "documents_completed_at",
    "approved_at",
    "activated_at",
    "rejected_at",
    "abandoned_at",
    "last_onboarding_event_type",
    "last_onboarding_event_at",
    "highest_stage_reached",
    "onboarding_status",
    "exit_from_stage",
    "onboarding_event_count",
    "days_since_registered",
    "days_in_current_stage",
    "is_commercial_activation_30d_eligible",
    "is_technically_activated",
    "days_to_technical_activation",
)


class ModeloOnboardingError(ValueError):
    """Indica que la entrada no es segura para construir el modelo BI."""


def _fecha(valor: str) -> datetime:
    """Convierte un timestamp ISO validado en datetime."""

    return datetime.fromisoformat(valor)


def _estado_onboarding(ultimo_evento: str) -> str:
    """Clasifica el proceso usando únicamente el último evento observado."""

    if ultimo_evento in {
        "ACTIVATED",
        "REJECTED",
        "ABANDONED",
    }:
        return ultimo_evento

    return "IN_PROGRESS"


def _etapa_mayor_alcanzada(
    tipos_evento: list[str],
) -> str:
    """Obtiene la etapa principal más avanzada de una ruta válida."""

    etapas_alcanzadas = [
        event_type
        for event_type in tipos_evento
        if event_type in ORDEN_ETAPAS
    ]

    return max(
        etapas_alcanzadas,
        key=ORDEN_ETAPAS.__getitem__,
    )


def _etapa_de_salida(
    tipos_evento: list[str],
) -> str:
    """Indica desde qué etapa principal ocurrió un rechazo o abandono."""

    if tipos_evento[-1] not in EVENTOS_DE_SALIDA:
        return ""

    return tipos_evento[-2]


def _validar_entradas(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impide construir BI con rutas rechazadas o datos parciales."""

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    if not resultado.violaciones.empty:
        reglas = ", ".join(
            sorted(
                set(
                    resultado.violaciones[
                        "rule_id"
                    ]
                )
            )
        )
        raise ModeloOnboardingError(
            "No se puede construir el modelo BI porque "
            f"la entrada contiene violaciones: {reglas}."
        )

    return (
        resultado.comercios_validos,
        resultado.eventos_validos,
    )


def construir_modelo_onboarding(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Convierte el historial de eventos en una fila por comercio."""

    comercios_validos, eventos_validos = (
        _validar_entradas(
            comercios,
            eventos,
            config,
        )
    )

    eventos_ordenados = eventos_validos.copy()
    eventos_ordenados["__event_datetime"] = (
        eventos_ordenados["event_at"].map(_fecha)
    )
    eventos_ordenados = eventos_ordenados.sort_values(
        [
            "merchant_id",
            "__event_datetime",
            "onboarding_event_id",
        ],
        kind="stable",
    )

    eventos_por_comercio = {
        str(merchant_id): grupo.reset_index(
            drop=True
        )
        for merchant_id, grupo in (
            eventos_ordenados.groupby(
                "merchant_id",
                sort=False,
            )
        )
    }

    filas: list[dict[str, object]] = []

    for _, comercio in (
        comercios_validos.sort_values(
            "merchant_id",
            kind="stable",
        ).iterrows()
    ):
        merchant_id = str(
            comercio["merchant_id"]
        )
        ruta = eventos_por_comercio[merchant_id]
        tipos_evento = ruta["event_type"].tolist()
        evento_por_tipo = {
            str(fila["event_type"]): str(
                fila["event_at"]
            )
            for _, fila in ruta.iterrows()
        }

        registered_at = evento_por_tipo["REGISTERED"]
        registered_date = _fecha(
            registered_at
        ).date()

        ultima_fila = ruta.iloc[-1]
        ultimo_evento = str(
            ultima_fila["event_type"]
        )
        ultimo_evento_at = str(
            ultima_fila["event_at"]
        )

        activated_at = evento_por_tipo.get(
            "ACTIVATED",
            "",
        )
        is_technically_activated = bool(
            activated_at
        )

        days_to_technical_activation: (
            int | None
        ) = None

        if is_technically_activated:
            days_to_technical_activation = (
                _fecha(activated_at).date()
                - registered_date
            ).days
            
        days_since_registered = (
            config.project.as_of_date
            - registered_date
        ).days

        days_observed_since_registered = (
            config.project.end_date
            - registered_date
        ).days

        days_in_current_stage = (
            config.project.as_of_date
            - _fecha(ultimo_evento_at).date()
        ).days

        filas.append(
            {
                "merchant_id": merchant_id,
                "merchant_name": comercio[
                    "merchant_name"
                ],
                "city": comercio["city"],
                "business_segment": comercio[
                    "business_segment"
                ],
                "acquisition_channel": comercio[
                    "acquisition_channel"
                ],
                "registered_cohort_month": (
                    registered_date.strftime(
                        "%Y-%m"
                    )
                ),
                "registered_at": registered_at,
                "validation_started_at": (
                    evento_por_tipo.get(
                        "VALIDATION_STARTED",
                        "",
                    )
                ),
                "documents_completed_at": (
                    evento_por_tipo.get(
                        "DOCUMENTS_COMPLETED",
                        "",
                    )
                ),
                "approved_at": evento_por_tipo.get(
                    "APPROVED",
                    "",
                ),
                "activated_at": activated_at,
                "rejected_at": evento_por_tipo.get(
                    "REJECTED",
                    "",
                ),
                "abandoned_at": evento_por_tipo.get(
                    "ABANDONED",
                    "",
                ),
                "last_onboarding_event_type": (
                    ultimo_evento
                ),
                "last_onboarding_event_at": (
                    ultimo_evento_at
                ),
                "highest_stage_reached": (
                    _etapa_mayor_alcanzada(
                        tipos_evento
                    )
                ),
                "onboarding_status": (
                    _estado_onboarding(
                        ultimo_evento
                    )
                ),
                "exit_from_stage": (
                    _etapa_de_salida(
                        tipos_evento
                    )
                ),
                "onboarding_event_count": len(
                    ruta
                ),
                "days_since_registered": (
                    days_since_registered
                ),
                "days_in_current_stage": (
                    days_in_current_stage
                ),
                (
                    "is_commercial_activation_30d_eligible"
                ): (
                    days_observed_since_registered
                    >= config.onboarding.first_transaction_window_days
                ),
                "is_technically_activated": (
                    is_technically_activated
                ),
                "days_to_technical_activation": (
                    days_to_technical_activation
                ),
            }
        )

    modelo = pd.DataFrame(
        filas,
        columns=COLUMNAS_MODELO_ONBOARDING,
    )

    if not modelo["merchant_id"].is_unique:
        raise AssertionError(
            "merchant_onboarding_bi.csv contiene "
            "merchant_id duplicados."
        )

    if len(modelo) != len(comercios_validos):
        raise AssertionError(
            "La conciliación del modelo por comercio "
            "no cierra."
        )

    return modelo


def guardar_modelo_onboarding(
    modelo: pd.DataFrame,
    ruta: str | Path = Path(
        "data/procesada/merchant_onboarding_bi.csv"
    ),
) -> Path:
    """Guarda la tabla BI de onboarding de forma reproducible."""

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
    """Construye el modelo usando las salidas procesadas."""

    config = load_config()
    comercios = pd.read_csv(
        "data/procesada/merchants.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    eventos = pd.read_csv(
        "data/procesada/onboarding_events.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    modelo = construir_modelo_onboarding(
        comercios,
        eventos,
        config,
    )
    ruta = guardar_modelo_onboarding(modelo)

    print(
        f"Modelo de onboarding generado: "
        f"{len(modelo)} comercios en {ruta}."
    )


if __name__ == "__main__":
    main()
