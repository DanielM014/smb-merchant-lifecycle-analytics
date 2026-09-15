from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.ventas.modelo import (
    COLUMNAS_MODELO_CICLO_VIDA,
    ModeloCicloVidaError,
    _actividad_posterior_30d,
    _estado_primera_transaccion,
    construir_modelo_ciclo_vida,
    guardar_modelo_ciclo_vida,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


@pytest.fixture(scope="module")
def datos_oficiales() -> tuple[
    ProjectConfig,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    config = load_config(CONFIG_PATH)
    modelo_onboarding = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "procesada"
        / "merchant_onboarding_bi.csv",
        dtype=str,
        keep_default_na=False,
    )
    ventas = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "procesada"
        / "merchant_sales_daily.csv",
        dtype=str,
        keep_default_na=False,
    )
    modelo = construir_modelo_ciclo_vida(
        modelo_onboarding,
        ventas,
        config,
    )

    return config, modelo_onboarding, ventas, modelo


def _fila(
    modelo: pd.DataFrame,
    merchant_id: str,
) -> pd.Series:
    return modelo.loc[
        modelo["merchant_id"].eq(merchant_id)
    ].iloc[0]


def test_modelo_oficial_conserva_grano_y_estados(
    datos_oficiales,
) -> None:
    _, _, _, modelo = datos_oficiales

    assert list(modelo.columns) == list(
        COLUMNAS_MODELO_CICLO_VIDA
    )
    assert len(modelo) == 180
    assert modelo["merchant_id"].is_unique
    assert modelo[
        "first_transaction_status"
    ].value_counts().to_dict() == {
        "ON_TIME": 84,
        "NOT_APPLICABLE": 82,
        "LATE": 9,
        "NOT_OBSERVED": 5,
    }


def test_anclas_distinguen_conversion_oportuna_tardia_y_ausente(
    datos_oficiales,
) -> None:
    _, _, _, modelo = datos_oficiales
    cafe_horizonte = _fila(modelo, "M001")
    conversion_tardia = _fila(modelo, "M009")
    sin_primera_transaccion = _fila(modelo, "M010")

    assert cafe_horizonte["first_transaction_date"] == "2025-01-14"
    assert cafe_horizonte["days_to_first_transaction"] == 13
    assert cafe_horizonte["first_transaction_status"] == "ON_TIME"
    assert bool(cafe_horizonte["commercial_activation_30d"])

    assert conversion_tardia["first_transaction_date"] == "2026-08-25"
    assert conversion_tardia["days_to_first_transaction"] == 32
    assert conversion_tardia["first_transaction_status"] == "LATE"
    assert not bool(conversion_tardia["commercial_activation_30d"])

    assert sin_primera_transaccion["first_transaction_date"] == ""
    assert pd.isna(
        sin_primera_transaccion["days_to_first_transaction"]
    )
    assert (
        sin_primera_transaccion["first_transaction_status"]
        == "NOT_OBSERVED"
    )
    assert pd.isna(
        sin_primera_transaccion["commercial_activation_30d"]
    )


def test_resultado_comercial_solo_existe_para_elegibles(
    datos_oficiales,
) -> None:
    _, _, _, modelo = datos_oficiales
    elegibles = modelo[
        "is_commercial_activation_30d_eligible"
    ]

    assert int(elegibles.sum()) == 168
    assert int(
        modelo["commercial_activation_30d"].sum()
    ) == 80
    assert modelo.loc[
        elegibles,
        "commercial_activation_30d",
    ].notna().all()
    assert modelo.loc[
        ~elegibles,
        "commercial_activation_30d",
    ].isna().all()


def test_active_30d_exige_ventana_posterior_completa(
    datos_oficiales,
) -> None:
    _, _, _, modelo = datos_oficiales
    elegibles = modelo["is_active_30d_eligible"]

    assert int(elegibles.sum()) == 87
    assert int(modelo["is_active_30d"].sum()) == 87
    assert modelo.loc[
        elegibles,
        "is_active_30d",
    ].notna().all()
    assert modelo.loc[
        ~elegibles,
        "is_active_30d",
    ].isna().all()


def test_dia_30_se_incluye_en_ambas_ventanas() -> None:
    assert (
        _estado_primera_transaccion(True, 30, 30)
        == "ON_TIME"
    )
    assert (
        _estado_primera_transaccion(True, 31, 30)
        == "LATE"
    )

    serie = pd.DataFrame(
        {
            "__sales_date": [
                date(2025, 1, 1),
                date(2025, 1, 31),
                date(2025, 2, 1),
            ],
            "approved_transactions": [1, 2, 3],
        }
    )

    assert _actividad_posterior_30d(
        serie,
        date(2025, 1, 1),
        30,
    )


def test_first_transaction_se_deriva_de_ventas_positivas(
    datos_oficiales,
) -> None:
    _, _, ventas, modelo = datos_oficiales
    esperadas = (
        ventas.loc[
            pd.to_numeric(
                ventas["approved_transactions"]
            ).gt(0)
        ]
        .groupby("merchant_id")["sales_date"]
        .min()
        .to_dict()
    )
    observadas = (
        modelo.loc[
            modelo["first_transaction_date"].ne("")
        ]
        .set_index("merchant_id")["first_transaction_date"]
        .to_dict()
    )

    assert observadas == esperadas


def test_rechaza_elegibilidad_desactualizada(
    datos_oficiales,
) -> None:
    config, modelo_onboarding, ventas, _ = datos_oficiales
    alterado = modelo_onboarding.copy(deep=True)
    actual = alterado.loc[
        0,
        "is_commercial_activation_30d_eligible",
    ]
    alterado.loc[
        0,
        "is_commercial_activation_30d_eligible",
    ] = "False" if actual == "True" else "True"

    with pytest.raises(
        ModeloCicloVidaError,
        match="última fecha observada",
    ):
        construir_modelo_ciclo_vida(
            alterado,
            ventas,
            config,
        )


def test_rechaza_ventas_que_no_superan_calidad(
    datos_oficiales,
) -> None:
    config, modelo_onboarding, ventas, _ = datos_oficiales
    indice = ventas.index[
        ventas["merchant_id"].eq("M001")
    ][0]
    incompletas = ventas.drop(index=indice).reset_index(
        drop=True
    )

    with pytest.raises(
        ModeloCicloVidaError,
        match="SALES_DATE_MISSING",
    ):
        construir_modelo_ciclo_vida(
            modelo_onboarding,
            incompletas,
            config,
        )


def test_construccion_no_modifica_las_entradas(
    datos_oficiales,
) -> None:
    config, modelo_onboarding, ventas, _ = datos_oficiales
    onboarding_original = modelo_onboarding.copy(deep=True)
    ventas_originales = ventas.copy(deep=True)

    construir_modelo_ciclo_vida(
        modelo_onboarding,
        ventas,
        config,
    )

    pd.testing.assert_frame_equal(
        modelo_onboarding,
        onboarding_original,
    )
    pd.testing.assert_frame_equal(
        ventas,
        ventas_originales,
    )


def test_archivo_guardado_es_reproducible(
    datos_oficiales,
    tmp_path: Path,
) -> None:
    _, _, _, modelo = datos_oficiales
    primera = guardar_modelo_ciclo_vida(
        modelo,
        tmp_path / "primera.csv",
    )
    segunda = guardar_modelo_ciclo_vida(
        modelo,
        tmp_path / "segunda.csv",
    )

    assert primera.read_bytes() == segunda.read_bytes()

    guardado = pd.read_csv(
        primera,
        dtype=str,
        keep_default_na=False,
    )
    assert list(guardado.columns) == list(
        COLUMNAS_MODELO_CICLO_VIDA
    )
    assert len(guardado) == 180
