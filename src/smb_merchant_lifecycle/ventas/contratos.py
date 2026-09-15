"""Contrato estructural de la actividad comercial diaria."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from math import isfinite
from numbers import Integral, Real

import pandas as pd
import pandera.pandas as pa

from smb_merchant_lifecycle.config import ProjectConfig


COLUMNAS_VENTAS_DIARIAS = (
    "merchant_id",
    "sales_date",
    "approved_transactions",
    "sales_amount_cop",
)


class ContratoVentasError(ValueError):
    """Indica que las columnas no cumplen el contrato de ventas."""


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


def _fecha_venta_es_valida(
    valor: object,
    config: ProjectConfig,
) -> bool:
    """Comprueba formato ISO diario y periodo configurado."""

    if not isinstance(valor, str):
        return False

    texto = valor.strip()

    try:
        fecha_venta = date.fromisoformat(texto)
    except ValueError:
        return False

    return (
        fecha_venta.isoformat() == texto
        and config.project.start_date
        <= fecha_venta
        <= config.project.end_date
    )


def _es_entero_no_negativo(valor: object) -> bool:
    """Acepta enteros reales o su representación decimal canónica."""

    if isinstance(valor, bool) or pd.isna(valor):
        return False

    if isinstance(valor, Integral):
        return int(valor) >= 0

    if isinstance(valor, Real):
        numero = float(valor)
        return (
            isfinite(numero)
            and numero.is_integer()
            and numero >= 0
        )

    if not isinstance(valor, str):
        return False

    return re.fullmatch(
        r"0|[1-9]\d*",
        valor.strip(),
    ) is not None


def _serie_enteros_no_negativos(
    serie: pd.Series,
) -> pd.Series:
    """Evalúa medidas enteras no negativas sin truncarlas."""

    return serie.map(_es_entero_no_negativo)


def validar_estructura_ventas(
    ventas: pd.DataFrame,
) -> None:
    """Detiene la ejecución si faltan, sobran o se mueven columnas."""

    observadas = tuple(
        str(columna)
        for columna in ventas.columns
    )

    if observadas == COLUMNAS_VENTAS_DIARIAS:
        return

    faltantes = [
        columna
        for columna in COLUMNAS_VENTAS_DIARIAS
        if columna not in observadas
    ]
    adicionales = [
        columna
        for columna in observadas
        if columna not in COLUMNAS_VENTAS_DIARIAS
    ]
    repetidas = [
        columna
        for columna, cantidad
        in Counter(observadas).items()
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

    raise ContratoVentasError(
        "merchant_sales_daily.csv no cumple su contrato "
        "estructural: "
        + "; ".join(detalles)
        + "."
    )


def crear_contrato_ventas(
    config: ProjectConfig,
) -> pa.DataFrameSchema:
    """Crea el contrato de ventas dependiente del periodo."""

    return pa.DataFrameSchema(
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
                        error=(
                            "merchant_id debe usar el "
                            "formato M000"
                        ),
                    ),
                ],
                nullable=False,
            ),
            "sales_date": pa.Column(
                str,
                checks=[
                    pa.Check(
                        _serie_no_vacia,
                        error="sales_date no puede estar vacío",
                    ),
                    pa.Check(
                        lambda serie: serie.map(
                            lambda valor: _fecha_venta_es_valida(
                                valor,
                                config,
                            )
                        ),
                        error=(
                            "sales_date debe ser YYYY-MM-DD "
                            "y estar dentro del periodo"
                        ),
                    ),
                ],
                nullable=False,
            ),
            "approved_transactions": pa.Column(
                object,
                checks=pa.Check(
                    _serie_enteros_no_negativos,
                    error=(
                        "approved_transactions debe ser "
                        "un entero no negativo"
                    ),
                ),
                nullable=False,
                coerce=True,
            ),
            "sales_amount_cop": pa.Column(
                object,
                checks=pa.Check(
                    _serie_enteros_no_negativos,
                    error=(
                        "sales_amount_cop debe ser un "
                        "entero no negativo"
                    ),
                ),
                nullable=False,
                coerce=True,
            ),
        },
        strict=True,
        ordered=True,
        unique_column_names=True,
        name="merchant_sales_daily",
    )


def validar_contrato_ventas(
    ventas: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Valida la estructura y las reglas de fila de ventas."""

    validar_estructura_ventas(ventas)

    return crear_contrato_ventas(
        config
    ).validate(
        ventas,
        lazy=True,
    )
