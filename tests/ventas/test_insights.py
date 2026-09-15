from pathlib import Path

import pandas as pd
import pytest

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.ventas.insights import (
    COLUMNAS_INSIGHTS,
    InsightsComercialesError,
    _prioridad,
    construir_insights_comerciales,
    guardar_insights_comerciales,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


@pytest.fixture(scope="module")
def datos_oficiales() -> dict[str, object]:
    config = load_config(CONFIG_PATH)
    metricas = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "procesada"
        / "merchant_sales_28d_metrics.csv",
        dtype=str,
        keep_default_na=False,
    )
    metricas_originales = metricas.copy(deep=True)
    insights = construir_insights_comerciales(
        metricas,
        config,
    )

    return {
        "config": config,
        "metricas": metricas,
        "metricas_originales": metricas_originales,
        "insights": insights,
    }


def _fila(
    dataframe: pd.DataFrame,
    merchant_id: str,
) -> pd.Series:
    return dataframe.loc[
        dataframe["merchant_id"].eq(merchant_id)
    ].iloc[0]


def test_salida_oficial_conserva_esquema_y_grano(
    datos_oficiales,
) -> None:
    insights = datos_oficiales["insights"]

    assert list(insights.columns) == list(COLUMNAS_INSIGHTS)
    assert len(insights) == 39
    assert insights["insight_id"].is_unique
    assert not insights.duplicated(
        subset=["merchant_id", "as_of_date", "insight_type"]
    ).any()
    assert set(insights["as_of_date"]) == {"2026-09-01"}


def test_tipos_y_prioridades_concilian(
    datos_oficiales,
) -> None:
    insights = datos_oficiales["insights"]

    assert insights["insight_type"].value_counts().to_dict() == {
        "SALES_VOLUME_GROWTH_28D": 22,
        "SALES_VOLUME_DECLINE_28D": 17,
    }
    assert insights["priority"].value_counts().to_dict() == {
        "HIGH": 20,
        "MEDIUM": 11,
        "LOW": 8,
    }


@pytest.mark.parametrize(
    (
        "merchant_id",
        "insight_id",
        "insight_type",
        "priority",
        "baseline",
        "actual",
        "change_pct",
    ),
    [
        (
            "M001",
            "INS-20260901-M001-SVD28D",
            "SALES_VOLUME_DECLINE_28D",
            "LOW",
            50_000_000,
            36_000_000,
            -28.0,
        ),
        (
            "M006",
            "INS-20260901-M006-SVG28D",
            "SALES_VOLUME_GROWTH_28D",
            "MEDIUM",
            30_000_000,
            42_000_000,
            40.0,
        ),
        (
            "M007",
            "INS-20260901-M007-SVD28D",
            "SALES_VOLUME_DECLINE_28D",
            "HIGH",
            28_000_000,
            0,
            -100.0,
        ),
    ],
)
def test_anclas_generan_insight_auditable(
    datos_oficiales,
    merchant_id: str,
    insight_id: str,
    insight_type: str,
    priority: str,
    baseline: int,
    actual: int,
    change_pct: float,
) -> None:
    fila = _fila(datos_oficiales["insights"], merchant_id)

    assert fila["insight_id"] == insight_id
    assert fila["insight_type"] == insight_type
    assert fila["priority"] == priority
    assert fila["baseline_value_cop"] == baseline
    assert fila["current_value_cop"] == actual
    assert fila["change_pct"] == change_pct


def test_evidencia_de_cafe_horizonte_conserva_valores_y_ventanas(
    datos_oficiales,
) -> None:
    fila = _fila(datos_oficiales["insights"], "M001")

    assert fila["evidence_text"] == (
        "El volumen aprobado cayó 28 %: de 50.000.000 COP "
        "(2026-07-07 a 2026-08-03) a 36.000.000 COP "
        "(2026-08-04 a 2026-08-31)."
    )
    assert fila["recommendation_text"] == (
        "Contactar al comercio para entender si existe un problema "
        "operativo, estacional o de continuidad del servicio."
    )


def test_crecimiento_tiene_recomendacion_especifica(
    datos_oficiales,
) -> None:
    fila = _fila(datos_oficiales["insights"], "M006")

    assert fila["evidence_text"].startswith(
        "El volumen aprobado creció 40 %:"
    )
    assert fila["recommendation_text"] == (
        "Contactar al comercio para comprender el crecimiento observado "
        "y evaluar oportunidades de acompañamiento comercial."
    )


def test_estables_y_no_elegibles_no_generan_insight(
    datos_oficiales,
) -> None:
    ids_con_insight = set(datos_oficiales["insights"]["merchant_id"])

    assert "M008" not in ids_con_insight
    assert "M009" not in ids_con_insight
    assert "M010" not in ids_con_insight


def test_cada_insight_concilia_con_una_bandera_de_metricas(
    datos_oficiales,
) -> None:
    insights = datos_oficiales["insights"]
    metricas = datos_oficiales["metricas"].copy(deep=True)
    unidas = insights.merge(
        metricas,
        on=["merchant_id", "as_of_date"],
        how="left",
        validate="one_to_one",
        suffixes=("_insight", "_metrica"),
    )

    assert len(unidas) == len(insights)
    assert (
        pd.to_numeric(unidas["baseline_value_cop_insight"])
        == pd.to_numeric(unidas["baseline_value_cop_metrica"])
    ).all()
    assert (
        pd.to_numeric(unidas["current_value_cop_insight"])
        == pd.to_numeric(unidas["current_value_cop_metrica"])
    ).all()
    assert (
        pd.to_numeric(unidas["change_pct_insight"])
        == pd.to_numeric(unidas["change_pct_metrica"])
    ).all()
    assert (
        unidas["is_sales_volume_decline_28d"].eq("True")
        == unidas["insight_type"].eq("SALES_VOLUME_DECLINE_28D")
    ).all()
    assert (
        unidas["is_sales_volume_growth_28d"].eq("True")
        == unidas["insight_type"].eq("SALES_VOLUME_GROWTH_28D")
    ).all()


@pytest.mark.parametrize(
    ("change_pct", "esperada"),
    [
        (100.0, "HIGH"),
        (-50.0, "HIGH"),
        (49.999999, "MEDIUM"),
        (-35.0, "MEDIUM"),
        (34.999999, "LOW"),
        (20.0, "LOW"),
        (-20.0, "LOW"),
    ],
)
def test_prioridad_usa_magnitud_y_limites_inclusivos(
    datos_oficiales,
    change_pct: float,
    esperada: str,
) -> None:
    config = datos_oficiales["config"]

    assert _prioridad(change_pct, config) == esperada


def test_no_admite_insight_por_debajo_del_minimo(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]

    with pytest.raises(
        InsightsComercialesError,
        match="prioridad mínima",
    ):
        _prioridad(19.999999, config)


def test_rechaza_caida_y_crecimiento_simultaneos(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    metricas = datos_oficiales["metricas"].copy(deep=True)
    indice = metricas.index[metricas["merchant_id"].eq("M001")][0]
    metricas.loc[indice, "is_sales_volume_growth_28d"] = "True"

    with pytest.raises(
        InsightsComercialesError,
        match="simultáneamente",
    ):
        construir_insights_comerciales(metricas, config)


def test_rechaza_senal_para_comercio_no_elegible(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    metricas = datos_oficiales["metricas"].copy(deep=True)
    indice = metricas.index[metricas["merchant_id"].eq("M009")][0]
    metricas.loc[indice, "is_sales_volume_decline_28d"] = "True"

    with pytest.raises(
        InsightsComercialesError,
        match="no elegible",
    ):
        construir_insights_comerciales(metricas, config)


def test_rechaza_porcentaje_que_no_concilia(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    metricas = datos_oficiales["metricas"].copy(deep=True)
    indice = metricas.index[metricas["merchant_id"].eq("M001")][0]
    metricas.loc[indice, "change_pct"] = "-27.0"

    with pytest.raises(
        InsightsComercialesError,
        match="no concilia",
    ):
        construir_insights_comerciales(metricas, config)


def test_rechaza_clave_de_metricas_duplicada(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    metricas = datos_oficiales["metricas"]
    duplicadas = pd.concat(
        [metricas, metricas.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        InsightsComercialesError,
        match="debe ser única",
    ):
        construir_insights_comerciales(duplicadas, config)


def test_rechaza_esquema_incompleto(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    incompletas = datos_oficiales["metricas"].drop(
        columns="change_pct"
    )

    with pytest.raises(
        InsightsComercialesError,
        match="esquema esperado",
    ):
        construir_insights_comerciales(incompletas, config)


def test_orden_de_entrada_no_cambia_los_insights(
    datos_oficiales,
) -> None:
    config = datos_oficiales["config"]
    metricas = datos_oficiales["metricas"]
    desordenadas = metricas.sample(frac=1, random_state=3)
    reconstruidos = construir_insights_comerciales(
        desordenadas,
        config,
    )

    pd.testing.assert_frame_equal(
        reconstruidos,
        datos_oficiales["insights"],
    )


def test_construccion_no_modifica_las_metricas(
    datos_oficiales,
) -> None:
    pd.testing.assert_frame_equal(
        datos_oficiales["metricas"],
        datos_oficiales["metricas_originales"],
    )


def test_archivo_guardado_es_reproducible(
    datos_oficiales,
    tmp_path: Path,
) -> None:
    insights = datos_oficiales["insights"]
    primera = tmp_path / "primera.csv"
    segunda = tmp_path / "segunda.csv"

    guardar_insights_comerciales(insights, primera)
    guardar_insights_comerciales(insights, segunda)

    assert primera.read_bytes() == segunda.read_bytes()
