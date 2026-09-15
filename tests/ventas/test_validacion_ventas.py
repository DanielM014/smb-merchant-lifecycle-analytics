from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from smb_merchant_lifecycle.config import load_config
from smb_merchant_lifecycle.ventas.validacion import (
    guardar_resultado_validacion_ventas,
    validar_integridad_ventas,
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


def _modelo() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "merchant_id": "M001",
                "activated_at": "2025-01-01T09:00:00-05:00",
                "is_technically_activated": "True",
            },
            {
                "merchant_id": "M002",
                "activated_at": "",
                "is_technically_activated": "False",
            },
        ]
    )


def _ventas_densas() -> pd.DataFrame:
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
            {
                "merchant_id": "M001",
                "sales_date": "2025-01-03",
                "approved_transactions": "1",
                "sales_amount_cop": "50000",
            },
        ]
    )


def test_datos_oficiales_pasan_sin_rechazos() -> None:
    config = load_config(CONFIG_PATH)
    ventas = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "merchant_sales_daily.csv",
        dtype=str,
        keep_default_na=False,
    )
    modelo = pd.read_csv(
        PROJECT_ROOT / "data" / "procesada" / "merchant_onboarding_bi.csv",
        dtype=str,
        keep_default_na=False,
    )

    resultado = validar_integridad_ventas(
        ventas,
        modelo,
        config,
    )

    assert len(resultado.ventas_validas) == 26737
    assert resultado.ventas_validas["merchant_id"].nunique() == 98
    assert resultado.violaciones.empty
    assert resultado.registros_rechazados.empty


def test_ceros_observados_son_validos() -> None:
    resultado = validar_integridad_ventas(
        _ventas_densas(),
        _modelo(),
        _config_corta(),
    )

    assert len(resultado.ventas_validas) == 3
    assert resultado.ventas_validas.iloc[0]["approved_transactions"] == 0
    assert resultado.violaciones.empty


def test_inconsistencia_de_monto_cuarentena_la_serie() -> None:
    ventas = _ventas_densas()
    ventas.loc[1, "sales_amount_cop"] = "0"

    resultado = validar_integridad_ventas(
        ventas,
        _modelo(),
        _config_corta(),
    )

    assert resultado.ventas_validas.empty
    assert len(resultado.registros_rechazados) == 3
    assert {
        "SALES_ZERO_AMOUNT_INCONSISTENT",
        "SALES_DATE_MISSING",
        "SALES_SERIES_QUARANTINED",
    }.issubset(set(resultado.violaciones["rule_id"]))


def test_clave_duplicada_cuarentena_la_serie() -> None:
    ventas = pd.concat(
        [_ventas_densas(), _ventas_densas().iloc[[1]]],
        ignore_index=True,
    )

    resultado = validar_integridad_ventas(
        ventas,
        _modelo(),
        _config_corta(),
    )

    assert resultado.ventas_validas.empty
    assert len(resultado.registros_rechazados) == 4
    assert "SALES_GRAIN_DUPLICATE" in set(
        resultado.violaciones["rule_id"]
    )
    assert "SALES_DATE_MISSING" in set(
        resultado.violaciones["rule_id"]
    )


def test_huerfano_y_no_activado_se_rechazan() -> None:
    extras = pd.DataFrame(
        [
            {
                "merchant_id": "M999",
                "sales_date": "2025-01-01",
                "approved_transactions": "1",
                "sales_amount_cop": "50000",
            },
            {
                "merchant_id": "M002",
                "sales_date": "2025-01-01",
                "approved_transactions": "1",
                "sales_amount_cop": "50000",
            },
        ]
    )
    ventas = pd.concat(
        [_ventas_densas(), extras],
        ignore_index=True,
    )

    resultado = validar_integridad_ventas(
        ventas,
        _modelo(),
        _config_corta(),
    )

    assert len(resultado.ventas_validas) == 3
    assert len(resultado.registros_rechazados) == 2
    assert {
        "SALES_MERCHANT_NOT_FOUND",
        "SALES_MERCHANT_NOT_ACTIVATED",
    }.issubset(set(resultado.violaciones["rule_id"]))


def test_venta_anterior_a_activacion_se_rechaza() -> None:
    modelo = _modelo()
    modelo.loc[
        modelo["merchant_id"].eq("M001"),
        "activated_at",
    ] = "2025-01-02T09:00:00-05:00"

    resultado = validar_integridad_ventas(
        _ventas_densas(),
        modelo,
        _config_corta(),
    )

    assert len(resultado.ventas_validas) == 2
    assert len(resultado.registros_rechazados) == 1
    assert set(resultado.violaciones["rule_id"]) == {
        "SALES_BEFORE_ACTIVATION"
    }


def test_fecha_ausente_cuarentena_la_serie() -> None:
    ventas = _ventas_densas().drop(
        index=[1]
    ).reset_index(drop=True)

    resultado = validar_integridad_ventas(
        ventas,
        _modelo(),
        _config_corta(),
    )

    assert resultado.ventas_validas.empty
    assert len(resultado.registros_rechazados) == 2
    brechas = resultado.violaciones.loc[
        resultado.violaciones["rule_id"].eq(
            "SALES_DATE_MISSING"
        )
    ]
    assert len(brechas) == 1
    assert pd.isna(brechas.iloc[0]["source_row_number"])
    assert brechas.iloc[0]["record_id"] == "M001|2025-01-02"


def test_guardado_es_reproducible_y_concilia(
    tmp_path: Path,
) -> None:
    ventas = _ventas_densas()
    ventas.loc[1, "sales_amount_cop"] = "0"
    resultado = validar_integridad_ventas(
        ventas,
        _modelo(),
        _config_corta(),
    )

    primera = guardar_resultado_validacion_ventas(
        resultado,
        tmp_path / "primera" / "procesada",
        tmp_path / "primera" / "rechazada",
    )
    segunda = guardar_resultado_validacion_ventas(
        resultado,
        tmp_path / "segunda" / "procesada",
        tmp_path / "segunda" / "rechazada",
    )

    assert len(resultado.ventas_validas) + len(
        resultado.registros_rechazados
    ) == len(ventas)

    for clave in primera:
        assert primera[clave].read_bytes() == segunda[clave].read_bytes()
