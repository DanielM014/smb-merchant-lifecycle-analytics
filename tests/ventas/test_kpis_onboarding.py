from pathlib import Path

import pandas as pd
import pytest

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.ventas.kpis import (
    COLUMNAS_RESUMEN_KPI,
    KPI_IDS,
    ResumenKpiError,
    construir_resumen_kpi,
    guardar_resumen_kpi,
)
from smb_merchant_lifecycle.ventas.modelo import (
    construir_modelo_ciclo_vida,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


@pytest.fixture(scope="module")
def modelo_oficial() -> tuple[
    ProjectConfig,
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

    return config, modelo


def _fila_kpi(
    resumen: pd.DataFrame,
    kpi_id: str,
) -> pd.Series:
    return resumen.loc[
        resumen["kpi_id"].eq(kpi_id)
    ].iloc[0]


def test_resumen_tiene_un_registro_por_kpi(
    modelo_oficial,
) -> None:
    config, modelo = modelo_oficial
    resumen = construir_resumen_kpi(
        modelo,
        config,
    )

    assert list(resumen.columns) == list(
        COLUMNAS_RESUMEN_KPI
    )
    assert tuple(resumen["kpi_id"]) == KPI_IDS
    assert len(resumen) == 3
    assert resumen["kpi_id"].is_unique
    assert set(resumen["scope"]) == {"GLOBAL"}
    assert set(resumen["as_of_date"]) == {"2026-09-01"}
    assert set(resumen["window_days"]) == {30}


def test_kpi_oficiales_tienen_valores_esperados(
    modelo_oficial,
) -> None:
    config, modelo = modelo_oficial
    resumen = construir_resumen_kpi(
        modelo,
        config,
    )
    activacion = _fila_kpi(
        resumen,
        "COMMERCIAL_ACTIVATION_30D_RATE",
    )
    mediana = _fila_kpi(
        resumen,
        "MEDIAN_DAYS_TO_FIRST_TRANSACTION_30D",
    )
    actividad = _fila_kpi(
        resumen,
        "ACTIVE_30D_RATE",
    )

    assert activacion["numerator_count"] == 80
    assert activacion["denominator_count"] == 168
    assert activacion["kpi_value"] == pytest.approx(
        47.619048
    )
    assert mediana["population_count"] == 80
    assert mediana["kpi_value"] == 18.0
    assert actividad["numerator_count"] == 87
    assert actividad["denominator_count"] == 87
    assert actividad["kpi_value"] == 100.0
    assert set(resumen["kpi_status"]) == {"AVAILABLE"}


def test_kpi_concilian_con_modelo_por_comercio(
    modelo_oficial,
) -> None:
    config, modelo = modelo_oficial
    resumen = construir_resumen_kpi(
        modelo,
        config,
    )
    activacion = _fila_kpi(
        resumen,
        "COMMERCIAL_ACTIVATION_30D_RATE",
    )
    actividad = _fila_kpi(
        resumen,
        "ACTIVE_30D_RATE",
    )

    assert activacion["numerator_count"] == int(
        modelo["commercial_activation_30d"].sum()
    )
    assert activacion["denominator_count"] == int(
        modelo[
            "is_commercial_activation_30d_eligible"
        ].sum()
    )
    assert actividad["numerator_count"] == int(
        modelo["is_active_30d"].sum()
    )
    assert actividad["denominator_count"] == int(
        modelo["is_active_30d_eligible"].sum()
    )


def test_denominadores_vacios_no_producen_division_invalida(
    modelo_oficial,
) -> None:
    config, modelo = modelo_oficial
    sin_observacion = modelo.copy(deep=True)
    sin_observacion[
        "is_commercial_activation_30d_eligible"
    ] = False
    sin_observacion["commercial_activation_30d"] = pd.array(
        [None] * len(sin_observacion),
        dtype="boolean",
    )
    sin_observacion["is_active_30d_eligible"] = False
    sin_observacion["is_active_30d"] = pd.array(
        [None] * len(sin_observacion),
        dtype="boolean",
    )
    resumen = construir_resumen_kpi(
        sin_observacion,
        config,
    )

    assert resumen["kpi_value"].isna().all()
    assert set(resumen["kpi_status"]) == {"NOT_AVAILABLE"}
    assert set(resumen["population_count"]) == {0}


def test_rechaza_resultado_en_comercio_no_elegible(
    modelo_oficial,
) -> None:
    config, modelo = modelo_oficial
    alterado = modelo.copy(deep=True)
    indice = alterado.index[
        ~alterado[
            "is_commercial_activation_30d_eligible"
        ]
    ][0]
    alterado.loc[
        indice,
        "commercial_activation_30d",
    ] = False

    with pytest.raises(
        ResumenKpiError,
        match="no elegible",
    ):
        construir_resumen_kpi(
            alterado,
            config,
        )


def test_archivo_kpi_guardado_es_reproducible(
    modelo_oficial,
    tmp_path: Path,
) -> None:
    config, modelo = modelo_oficial
    resumen = construir_resumen_kpi(
        modelo,
        config,
    )
    primera = guardar_resumen_kpi(
        resumen,
        tmp_path / "primera.csv",
    )
    segunda = guardar_resumen_kpi(
        resumen,
        tmp_path / "segunda.csv",
    )

    assert primera.read_bytes() == segunda.read_bytes()

    guardado = pd.read_csv(
        primera,
        dtype=str,
        keep_default_na=False,
    )
    assert list(guardado.columns) == list(
        COLUMNAS_RESUMEN_KPI
    )
    assert len(guardado) == 3
