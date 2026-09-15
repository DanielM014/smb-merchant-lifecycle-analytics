"""Contratos estructurales de los datos de comercios y onboarding."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import pandera.pandas as pa

from smb_merchant_lifecycle.config import ProjectConfig
from smb_merchant_lifecycle.generacion_comercios import (
    CANALES_ADQUISICION,
    CIUDADES,
    SEGMENTOS,
)
from smb_merchant_lifecycle.generacion_onboarding import (
    TRANSICIONES_PERMITIDAS,
)


COLUMNAS_COMERCIOS = (
    "merchant_id",
    "merchant_name",
    "city",
    "business_segment",
    "acquisition_channel",
)

COLUMNAS_EVENTOS_ONBOARDING = (
    "onboarding_event_id",
    "merchant_id",
    "event_type",
    "event_at",
)

DESPLAZAMIENTO_COLOMBIA = timedelta(hours=-5)


class ContratoEstructuralError(ValueError):
    """Indica que las columnas no cumplen el contrato de una tabla."""


def _serie_no_vacia(serie: pd.Series) -> pd.Series:
    """Marca como válidos los textos que contienen caracteres útiles."""

    return (
        serie.astype("string")
        .str.strip()
        .str.len()
        .gt(0)
        .fillna(False)
    )


def _serie_cumple_patron(
    serie: pd.Series,
    patron: str,
) -> pd.Series:
    """Evalúa un patrón completo sin aceptar valores nulos."""

    return (
        serie.astype("string")
        .str.fullmatch(patron)
        .fillna(False)
    )


def _validar_columnas_exactas(
    datos: pd.DataFrame,
    columnas_esperadas: tuple[str, ...],
    nombre_tabla: str,
) -> None:
    """Detiene la ejecución si faltan, sobran o se mueven columnas."""

    columnas_observadas = tuple(
        str(columna)
        for columna in datos.columns
    )

    if columnas_observadas == columnas_esperadas:
        return

    faltantes = [
        columna
        for columna in columnas_esperadas
        if columna not in columnas_observadas
    ]

    adicionales = [
        columna
        for columna in columnas_observadas
        if columna not in columnas_esperadas
    ]

    repetidas = [
        columna
        for columna, cantidad
        in Counter(columnas_observadas).items()
        if cantidad > 1
    ]

    detalles: list[str] = []

    if faltantes:
        detalles.append(
            "faltantes=" + ", ".join(faltantes)
        )

    if adicionales:
        detalles.append(
            "adicionales=" + ", ".join(adicionales)
        )

    if repetidas:
        detalles.append(
            "repetidas=" + ", ".join(repetidas)
        )

    if not faltantes and not adicionales and not repetidas:
        detalles.append("orden de columnas inválido")

    raise ContratoEstructuralError(
        f"{nombre_tabla} no cumple su contrato estructural: "
        + "; ".join(detalles)
        + "."
    )


def _event_at_es_valido(
    valor: object,
    config: ProjectConfig,
) -> bool:
    """Comprueba formato ISO, zona colombiana y periodo configurado."""

    if not isinstance(valor, str):
        return False

    try:
        fecha_evento = datetime.fromisoformat(
            valor
        )
    except ValueError:
        return False

    if (
        fecha_evento.tzinfo is None
        or fecha_evento.utcoffset()
        != DESPLAZAMIENTO_COLOMBIA
    ):
        return False

    return (
        config.project.start_date
        <= fecha_evento.date()
        <= config.project.end_date
    )


CONTRATO_COMERCIOS = pa.DataFrameSchema(
    {
        "merchant_id": pa.Column(
            str,
            checks=[
                pa.Check(
                    _serie_no_vacia,
                    error="merchant_id no puede estar vacío",
                ),
                pa.Check(
                    lambda serie: _serie_cumple_patron(
                        serie,
                        r"M\d{3}",
                    ),
                    error="merchant_id debe usar el formato M000",
                ),
            ],
            nullable=False,
            unique=True,
            report_duplicates="all",
        ),
        "merchant_name": pa.Column(
            str,
            checks=pa.Check(
                _serie_no_vacia,
                error="merchant_name no puede estar vacío",
            ),
            nullable=False,
        ),
        "city": pa.Column(
            str,
            checks=pa.Check.isin(CIUDADES),
            nullable=False,
        ),
        "business_segment": pa.Column(
            str,
            checks=pa.Check.isin(SEGMENTOS),
            nullable=False,
        ),
        "acquisition_channel": pa.Column(
            str,
            checks=pa.Check.isin(
                CANALES_ADQUISICION
            ),
            nullable=False,
        ),
    },
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="merchants",
)


def crear_contrato_eventos_onboarding(
    config: ProjectConfig,
) -> pa.DataFrameSchema:
    """Crea el contrato de eventos dependiente del periodo configurado."""

    return pa.DataFrameSchema(
        {
            "onboarding_event_id": pa.Column(
                str,
                checks=[
                    pa.Check(
                        _serie_no_vacia,
                        error=(
                            "onboarding_event_id no puede "
                            "estar vacío"
                        ),
                    ),
                    pa.Check(
                        lambda serie: (
                            _serie_cumple_patron(
                                serie,
                                r"OE\d{4}",
                            )
                        ),
                        error=(
                            "onboarding_event_id debe usar "
                            "el formato OE0000"
                        ),
                    ),
                ],
                nullable=False,
                unique=True,
                report_duplicates="all",
            ),
            "merchant_id": pa.Column(
                str,
                checks=[
                    pa.Check(
                        _serie_no_vacia,
                        error="merchant_id no puede estar vacío",
                    ),
                    pa.Check(
                        lambda serie: (
                            _serie_cumple_patron(
                                serie,
                                r"M\d{3}",
                            )
                        ),
                        error="merchant_id debe usar el formato M000",
                    ),
                ],
                nullable=False,
            ),
            "event_type": pa.Column(
                str,
                checks=pa.Check.isin(
                    tuple(TRANSICIONES_PERMITIDAS)
                ),
                nullable=False,
            ),
            "event_at": pa.Column(
                str,
                checks=[
                    pa.Check(
                        _serie_no_vacia,
                        error="event_at no puede estar vacío",
                    ),
                    pa.Check(
                        lambda serie: serie.map(
                            lambda valor: (
                                _event_at_es_valido(
                                    valor,
                                    config,
                                )
                            )
                        ),
                        error=(
                            "event_at debe ser ISO 8601, usar "
                            "UTC-05:00 y estar dentro del periodo"
                        ),
                    ),
                ],
                nullable=False,
            ),
        },
        strict=True,
        ordered=True,
        unique_column_names=True,
        name="onboarding_events",
    )


def validar_contrato_comercios(
    comercios: pd.DataFrame,
) -> pd.DataFrame:
    """Valida la estructura y las reglas de fila de comercios."""

    _validar_columnas_exactas(
        comercios,
        COLUMNAS_COMERCIOS,
        "merchants.csv",
    )

    return CONTRATO_COMERCIOS.validate(
        comercios,
        lazy=True,
    )


def validar_contrato_eventos_onboarding(
    eventos: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Valida la estructura y las reglas de fila de onboarding."""

    _validar_columnas_exactas(
        eventos,
        COLUMNAS_EVENTOS_ONBOARDING,
        "onboarding_events.csv",
    )

    contrato = crear_contrato_eventos_onboarding(
        config
    )

    return contrato.validate(
        eventos,
        lazy=True,
    )