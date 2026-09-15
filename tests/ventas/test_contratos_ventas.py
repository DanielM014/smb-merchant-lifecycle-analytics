from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from smb_merchant_lifecycle.config import load_config
from smb_merchant_lifecycle.ventas.contratos import (
    ContratoVentasError,
    validar_contrato_ventas,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


def _config_corta():
    config = load_config(CONFIG_PATH)
    return replace(
        config,
        project=replace(
            config.project,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 3),
            as_of_date=date(2025, 1, 4),
        ),
    )


def _ventas_validas() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "merchant_id": "M001",
                "sales_date": "2025-01-01",
                "approved_transactions": "0",
                "sales_amount_cop": "0",
            },
            {
                "merchant_id": "M001",
                "sales_date": "2025-01-02",
                "approved_transactions": "2",
                "sales_amount_cop": "100000",
            },
        ]
    )


def test_contrato_acepta_ventas_validas() -> None:
    resultado = validar_contrato_ventas(
        _ventas_validas(),
        _config_corta(),
    )

    assert len(resultado) == 2


def test_contrato_rechaza_columna_faltante() -> None:
    with pytest.raises(
        ContratoVentasError,
        match="faltantes=sales_amount_cop",
    ):
        validar_contrato_ventas(
            _ventas_validas().drop(
                columns=["sales_amount_cop"]
            ),
            _config_corta(),
        )


def test_contrato_rechaza_columna_extra_y_orden() -> None:
    ventas = _ventas_validas()

    with pytest.raises(
        ContratoVentasError,
        match="adicionales=currency",
    ):
        validar_contrato_ventas(
            ventas.assign(currency="COP"),
            _config_corta(),
        )

    with pytest.raises(
        ContratoVentasError,
        match="orden de columnas inválido",
    ):
        validar_contrato_ventas(
            ventas.loc[:, list(reversed(ventas.columns))],
            _config_corta(),
        )


def test_contrato_rechaza_fecha_invalida() -> None:
    ventas = _ventas_validas()
    ventas.loc[0, "sales_date"] = "2025/01/01"

    with pytest.raises(pa.errors.SchemaErrors):
        validar_contrato_ventas(
            ventas,
            _config_corta(),
        )


def test_contrato_rechaza_medidas_invalidas() -> None:
    ventas = _ventas_validas()
    ventas.loc[0, "approved_transactions"] = "-1"
    ventas.loc[1, "sales_amount_cop"] = "1.5"

    with pytest.raises(pa.errors.SchemaErrors) as error:
        validar_contrato_ventas(
            ventas,
            _config_corta(),
        )

    columnas = set(error.value.failure_cases["column"])
    assert columnas == {
        "approved_transactions",
        "sales_amount_cop",
    }
