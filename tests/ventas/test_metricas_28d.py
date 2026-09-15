from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.ventas.metricas_28d import (
    COLUMNAS_METRICAS_28D,
    MetricasVentas28dError,
    _calcular_change_pct,
    _fechas_ventanas,
    _motivo_no_elegibilidad,
    construir_metricas_28d,
    guardar_metricas_28d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


@pytest.fixture(scope="module")
def datos_oficiales() -> dict[str, object]:
    config = load_config(CONFIG_PATH)
    modelo = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "procesada"
        / "merchant_lifecycle_bi.csv",
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
    modelo_original = modelo.copy(deep=True)
    ventas_originales = ventas.copy(deep=True)
    metricas = construir_metricas_28d(
        modelo,
        ventas,
        config,
    )

    return {
        "config": config,
        "modelo": modelo,
        "ventas": ventas,
        "modelo_original": modelo_original,
        "ventas_originales": ventas_originales,
        "metricas": metricas,
    }


def _fila(
    metricas: pd.DataFrame,
    merchant_id: str,
) -> pd.Series:
    return metricas.loc[
        metricas["merchant_id"].eq(merchant_id)
    ].iloc[0]


def _evaluar_motivo(
    config: ProjectConfig,
    **cambios: object,
) -> str:
    baseline_start, _, _, _ = _fechas_ventanas(config)
    argumentos: dict[str, object] = {
        "first_transaction_date": baseline_start,
        "baseline_start": baseline_start,
        "ventanas_completas": True,
        "baseline_value_cop": 100,
        "baseline_approved_transactions": 10,
        "baseline_active_days": 7,
        "config": config,
    }
    argumentos.update(cambios)

    return _motivo_no_elegibilidad(**argumentos)


def test_salida_oficial_conserva_grano_esquema_y_corte(
    datos_oficiales,
) -> None:
    metricas = datos_oficiales["metricas"]

    assert list(metricas.columns) == list(COLUMNAS_METRICAS_28D)
    assert len(metricas) == 180
    assert not metricas.duplicated(
        subset=["merchant_id", "as_of_date"]
    ).any()
    assert set(metricas["as_of_date"]) == {"2026-09-01"}
    assert set(metricas["baseline_start"]) == {"2026-07-07"}
    assert set(metricas["baseline_end"]) == {"2026-08-03"}
    assert set(metricas["observation_start"]) == {"2026-08-04"}
    assert set(metricas["observation_end"]) == {"2026-08-31"}


def test_distribucion_oficial_concilia_con_anclas_sinteticas(
    datos_oficiales,
) -> None:
    metricas = datos_oficiales["metricas"]
    elegibles = metricas["is_sales_change_eligible"]
    caidas = metricas["is_sales_volume_decline_28d"]
    crecimientos = metricas["is_sales_volume_growth_28d"]

    assert int(elegibles.sum()) == 83
    assert int(caidas.sum()) == 17
    assert int(crecimientos.sum()) == 22
    assert int((elegibles & ~caidas & ~crecimientos).sum()) == 44
    assert metricas[
        "sales_change_ineligibility_reason"
    ].value_counts().to_dict() == {
        "INSUFFICIENT_HISTORY": 97,
        "": 83,
    }


@pytest.mark.parametrize(
    (
        "merchant_id",
        "baseline",
        "actual",
        "change_pct",
        "es_caida",
        "es_crecimiento",
    ),
    [
        ("M001", 50_000_000, 36_000_000, -28.0, True, False),
        ("M006", 30_000_000, 42_000_000, 40.0, False, True),
        ("M007", 28_000_000, 0, -100.0, True, False),
        ("M008", 40_000_000, 42_000_000, 5.0, False, False),
    ],
)
def test_anclas_direccionales(
    datos_oficiales,
    merchant_id: str,
    baseline: int,
    actual: int,
    change_pct: float,
    es_caida: bool,
    es_crecimiento: bool,
) -> None:
    fila = _fila(datos_oficiales["metricas"], merchant_id)

    assert fila["baseline_value_cop"] == baseline
    assert fila["current_value_cop"] == actual
    assert fila["change_pct"] == change_pct
    assert bool(fila["is_sales_volume_decline_28d"]) is es_caida
    assert bool(fila["is_sales_volume_growth_28d"]) is es_crecimiento


def test_sin_historia_no_se_confunde_con_cambio_cero(
    datos_oficiales,
) -> None:
    metricas = datos_oficiales["metricas"]

    for merchant_id in ("M009", "M010"):
        fila = _fila(metricas, merchant_id)

        assert not bool(fila["is_sales_change_eligible"])
        assert (
            fila["sales_change_ineligibility_reason"]
            == "INSUFFICIENT_HISTORY"
        )
        assert pd.isna(fila["change_pct"])
        assert not bool(fila["is_sales_volume_decline_28d"])
        assert not bool(fila["is_sales_volume_growth_28d"])


def test_exactamente_56_dias_es_historia_suficiente(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]

    assert _evaluar_motivo(config) == ""


def test_un_dia_menos_es_historia_insuficiente(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    baseline_start, _, _, _ = _fechas_ventanas(config)

    assert _evaluar_motivo(
        config,
        first_transaction_date=(
            baseline_start + timedelta(days=1)
        ),
    ) == "INSUFFICIENT_HISTORY"


def test_orden_de_reglas_prioriza_historia_y_cobertura(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    baseline_start, _, _, _ = _fechas_ventanas(config)

    assert _evaluar_motivo(
        config,
        first_transaction_date=None,
        ventanas_completas=False,
        baseline_value_cop=None,
        baseline_approved_transactions=None,
        baseline_active_days=None,
    ) == "INSUFFICIENT_HISTORY"
    assert _evaluar_motivo(
        config,
        first_transaction_date=baseline_start,
        ventanas_completas=False,
        baseline_value_cop=None,
        baseline_approved_transactions=None,
        baseline_active_days=None,
    ) == "INCOMPLETE_WINDOW"


def test_linea_base_cero_no_produce_porcentaje(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]

    assert _evaluar_motivo(
        config,
        baseline_value_cop=0,
        baseline_approved_transactions=0,
        baseline_active_days=0,
    ) == "ZERO_BASELINE"


@pytest.mark.parametrize(
    ("transacciones", "dias_activos"),
    [
        (9, 7),
        (10, 6),
    ],
)
def test_actividad_minima_debe_cumplir_ambas_condiciones(
    datos_oficiales,
    transacciones: int,
    dias_activos: int,
) -> None:
    config = datos_oficiales["config"]

    assert _evaluar_motivo(
        config,
        baseline_approved_transactions=transacciones,
        baseline_active_days=dias_activos,
    ) == "INSUFFICIENT_BASELINE_ACTIVITY"


def test_umbrales_direccionales_son_inclusivos(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    caida = _calcular_change_pct(100, 80)
    crecimiento = _calcular_change_pct(100, 120)

    assert caida == config.sales_change.decline_threshold_pct
    assert crecimiento == config.sales_change.growth_threshold_pct
    assert caida <= config.sales_change.decline_threshold_pct
    assert crecimiento >= config.sales_change.growth_threshold_pct


def test_ventana_incompleta_no_se_convierte_en_cero(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    modelo = datos_oficiales["modelo"]
    ventas = datos_oficiales["ventas"]
    indice = ventas.index[
        ventas["merchant_id"].eq("M001")
        & ventas["sales_date"].eq("2026-08-31")
    ][0]
    incompletas = ventas.drop(index=indice).reset_index(drop=True)
    metricas = construir_metricas_28d(
        modelo,
        incompletas,
        config,
    )
    fila = _fila(metricas, "M001")

    assert not bool(fila["is_sales_change_eligible"])
    assert (
        fila["sales_change_ineligibility_reason"]
        == "INCOMPLETE_WINDOW"
    )
    assert pd.isna(fila["current_value_cop"])
    assert pd.isna(fila["change_pct"])
    assert not bool(fila["is_sales_volume_decline_28d"])


def test_rechaza_grano_diario_duplicado(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    modelo = datos_oficiales["modelo"]
    ventas = datos_oficiales["ventas"]
    duplicadas = pd.concat(
        [ventas, ventas.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        MetricasVentas28dError,
        match="debe ser única",
    ):
        construir_metricas_28d(
            modelo,
            duplicadas,
            config,
        )


def test_construccion_no_modifica_las_entradas(
    datos_oficiales,
) -> None:
    pd.testing.assert_frame_equal(
        datos_oficiales["modelo"],
        datos_oficiales["modelo_original"],
    )
    pd.testing.assert_frame_equal(
        datos_oficiales["ventas"],
        datos_oficiales["ventas_originales"],
    )


def test_archivo_guardado_es_reproducible(
    datos_oficiales,
    tmp_path: Path,
) -> None:
    metricas = datos_oficiales["metricas"]
    primera = tmp_path / "primera.csv"
    segunda = tmp_path / "segunda.csv"

    guardar_metricas_28d(metricas, primera)
    guardar_metricas_28d(metricas, segunda)

    assert primera.read_bytes() == segunda.read_bytes()
