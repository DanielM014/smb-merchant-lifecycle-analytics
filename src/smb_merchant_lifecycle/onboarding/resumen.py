"""Resumen reconciliado del funnel técnico de onboarding."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from smb_merchant_lifecycle.config import load_config


ETAPAS_FUNNEL = (
    ("REGISTERED", "Registrados", "registered_at"),
    (
        "VALIDATION_STARTED",
        "Validación iniciada",
        "validation_started_at",
    ),
    (
        "DOCUMENTS_COMPLETED",
        "Documentos completados",
        "documents_completed_at",
    ),
    ("APPROVED", "Aprobados", "approved_at"),
    ("ACTIVATED", "Activados", "activated_at"),
)

COLUMNAS_RESUMEN_FUNNEL = (
    "stage",
    "stage_label",
    "stage_order",
    "merchant_count",
    "previous_stage",
    "previous_stage_count",
    "conversion_from_previous_pct",
    "non_progression_from_previous_count",
    "non_progression_from_previous_pct",
    "as_of_date",
)


class ResumenFunnelError(ValueError):
    """Indica que el modelo no permite calcular un funnel confiable."""


def _tiene_valor(serie: pd.Series) -> pd.Series:
    """Detecta timestamps presentes incluso si llegan valores nulos."""

    return (
        serie.notna()
        & serie.astype(str).str.strip().ne("")
    )


def _validar_modelo(modelo: pd.DataFrame) -> None:
    """Valida las condiciones mínimas para resumir el funnel."""

    columnas_requeridas = {
        "merchant_id",
        *(
            columna
            for _, _, columna in ETAPAS_FUNNEL
        ),
    }
    columnas_faltantes = sorted(
        columnas_requeridas - set(modelo.columns)
    )

    if columnas_faltantes:
        raise ResumenFunnelError(
            "Faltan columnas requeridas en el modelo: "
            f"{', '.join(columnas_faltantes)}."
        )

    if modelo.empty:
        raise ResumenFunnelError(
            "El modelo de onboarding está vacío."
        )

    if not modelo["merchant_id"].is_unique:
        raise ResumenFunnelError(
            "merchant_id debe ser único en el modelo "
            "de onboarding."
        )


def construir_resumen_funnel(
    modelo: pd.DataFrame,
    as_of_date: date,
) -> pd.DataFrame:
    """Resume alcance, conversión y no progresión por etapa."""

    _validar_modelo(modelo)

    conteos = [
        int(
            _tiene_valor(
                modelo[columna]
            ).sum()
        )
        for _, _, columna in ETAPAS_FUNNEL
    ]

    if conteos[0] != len(modelo):
        raise ResumenFunnelError(
            "Todo comercio debe tener registered_at."
        )

    if any(
        conteo_actual > conteo_anterior
        for conteo_anterior, conteo_actual in zip(
            conteos,
            conteos[1:],
        )
    ):
        raise ResumenFunnelError(
            "Los conteos del funnel no son monotónicos."
        )

    filas: list[dict[str, object]] = []

    for indice, (
        stage,
        stage_label,
        _,
    ) in enumerate(ETAPAS_FUNNEL):
        merchant_count = conteos[indice]

        previous_stage: str = ""
        previous_stage_count: int | None = None
        conversion_pct: float | None = None
        non_progression_count: int | None = None
        non_progression_pct: float | None = None

        if indice > 0:
            previous_stage = ETAPAS_FUNNEL[
                indice - 1
            ][0]
            previous_stage_count = conteos[
                indice - 1
            ]

            non_progression_count = (
                previous_stage_count
                - merchant_count
            )

            if previous_stage_count > 0:
                conversion_pct = round(
                    100
                    * merchant_count
                    / previous_stage_count,
                    6,
                )

                non_progression_pct = round(
                    100
                    * non_progression_count
                    / previous_stage_count,
                    6,
                )

        filas.append(
            {
                "stage": stage,
                "stage_label": stage_label,
                "stage_order": indice + 1,
                "merchant_count": merchant_count,
                "previous_stage": previous_stage,
                "previous_stage_count": (
                    previous_stage_count
                ),
                "conversion_from_previous_pct": (
                    conversion_pct
                ),
                (
                    "non_progression_from_previous_count"
                ): non_progression_count,
                (
                    "non_progression_from_previous_pct"
                ): non_progression_pct,
                "as_of_date": as_of_date.isoformat(),
            }
        )

    resumen = pd.DataFrame(
        filas,
        columns=COLUMNAS_RESUMEN_FUNNEL,
    )

    for columna in (
        "previous_stage_count",
        "non_progression_from_previous_count",
    ):
        resumen[columna] = pd.array(
            resumen[columna],
            dtype="Int64",
        )

    return resumen


def guardar_resumen_funnel(
    resumen: pd.DataFrame,
    ruta: str | Path = Path(
        "data/procesada/"
        "resumen_funnel_onboarding.csv"
    ),
) -> Path:
    """Guarda el resumen con bytes reproducibles."""

    ruta_salida = Path(ruta)
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumen.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return ruta_salida


def main() -> None:
    """Construye el resumen desde el modelo procesado."""

    config = load_config()

    modelo = pd.read_csv(
        "data/procesada/merchant_onboarding_bi.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    resumen = construir_resumen_funnel(
        modelo,
        config.project.as_of_date,
    )

    ruta = guardar_resumen_funnel(
        resumen
    )

    print(
        "Resumen del funnel generado: "
        f"{len(resumen)} etapas en {ruta}."
    )


if __name__ == "__main__":
    main()