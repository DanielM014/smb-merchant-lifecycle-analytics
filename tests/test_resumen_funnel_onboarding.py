from pathlib import Path

import pandas as pd
import pytest

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.generacion_comercios import (
    generar_comercios,
)
from smb_merchant_lifecycle.generacion_onboarding import (
    generar_eventos_onboarding,
)
from smb_merchant_lifecycle.modelo_onboarding import (
    construir_modelo_onboarding,
)
from smb_merchant_lifecycle.resumen_funnel_onboarding import (
    COLUMNAS_RESUMEN_FUNNEL,
    ETAPAS_FUNNEL,
    ResumenFunnelError,
    construir_resumen_funnel,
    guardar_resumen_funnel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "project.toml"
)


def _generar_modelo() -> tuple[
    ProjectConfig,
    pd.DataFrame,
]:
    config = load_config(CONFIG_PATH)

    comercios = generar_comercios(
        config
    )

    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )

    modelo = construir_modelo_onboarding(
        comercios,
        eventos,
        config,
    )

    return config, modelo


def test_resumen_tiene_una_fila_por_etapa() -> None:
    config, modelo = _generar_modelo()

    resumen = construir_resumen_funnel(
        modelo,
        config.project.as_of_date,
    )

    assert list(resumen.columns) == list(
        COLUMNAS_RESUMEN_FUNNEL
    )

    assert resumen["stage"].tolist() == [
        etapa
        for etapa, _, _ in ETAPAS_FUNNEL
    ]

    assert resumen["stage_order"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_conteos_del_funnel_concilian() -> None:
    config, modelo = _generar_modelo()

    resumen = construir_resumen_funnel(
        modelo,
        config.project.as_of_date,
    )

    assert resumen[
        "merchant_count"
    ].tolist() == [
        180,
        172,
        141,
        106,
        98,
    ]


def test_conversiones_usan_la_etapa_anterior() -> None:
    config, modelo = _generar_modelo()

    resumen = construir_resumen_funnel(
        modelo,
        config.project.as_of_date,
    ).set_index("stage")

    assert pd.isna(
        resumen.loc[
            "REGISTERED",
            "conversion_from_previous_pct",
        ]
    )

    assert resumen.loc[
        "VALIDATION_STARTED",
        "conversion_from_previous_pct",
    ] == pytest.approx(
        95.555556
    )

    assert resumen.loc[
        "DOCUMENTS_COMPLETED",
        "conversion_from_previous_pct",
    ] == pytest.approx(
        81.976744
    )

    assert resumen.loc[
        "APPROVED",
        "conversion_from_previous_pct",
    ] == pytest.approx(
        75.177305
    )

    assert resumen.loc[
        "ACTIVATED",
        "conversion_from_previous_pct",
    ] == pytest.approx(
        92.452830
    )


def test_no_progresion_reconcilia_con_conversiones() -> None:
    config, modelo = _generar_modelo()

    resumen = construir_resumen_funnel(
        modelo,
        config.project.as_of_date,
    )

    transiciones = resumen.iloc[1:]

    assert transiciones[
        "non_progression_from_previous_count"
    ].tolist() == [
        8,
        31,
        35,
        8,
    ]

    assert (
        transiciones[
            "conversion_from_previous_pct"
        ]
        + transiciones[
            "non_progression_from_previous_pct"
        ]
    ).tolist() == pytest.approx(
        [
            100.0,
            100.0,
            100.0,
            100.0,
        ]
    )


def test_fecha_de_corte_es_explicita() -> None:
    config, modelo = _generar_modelo()

    resumen = construir_resumen_funnel(
        modelo,
        config.project.as_of_date,
    )

    assert set(
        resumen["as_of_date"]
    ) == {
        config.project.as_of_date.isoformat()
    }


def test_orden_del_modelo_no_cambia_el_resumen() -> None:
    config, modelo = _generar_modelo()

    esperado = construir_resumen_funnel(
        modelo,
        config.project.as_of_date,
    )

    observado = construir_resumen_funnel(
        modelo.sample(
            frac=1,
            random_state=14,
        ).reset_index(
            drop=True
        ),
        config.project.as_of_date,
    )

    pd.testing.assert_frame_equal(
        observado,
        esperado,
    )


def test_rechaza_columnas_faltantes() -> None:
    config, modelo = _generar_modelo()

    modelo = modelo.drop(
        columns=[
            "approved_at",
        ]
    )

    with pytest.raises(
        ResumenFunnelError,
        match="approved_at",
    ):
        construir_resumen_funnel(
            modelo,
            config.project.as_of_date,
        )


def test_rechaza_grano_duplicado() -> None:
    config, modelo = _generar_modelo()

    modelo = pd.concat(
        [
            modelo,
            modelo.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ResumenFunnelError,
        match="merchant_id debe ser único",
    ):
        construir_resumen_funnel(
            modelo,
            config.project.as_of_date,
        )


def test_rechaza_funnel_no_monotonico() -> None:
    config = load_config(
        CONFIG_PATH
    )

    modelo = pd.DataFrame(
        [
            {
                "merchant_id": "M001",
                "registered_at": (
                    "2026-01-01T00:00:00-05:00"
                ),
                "validation_started_at": "",
                "documents_completed_at": (
                    "2026-01-02T00:00:00-05:00"
                ),
                "approved_at": "",
                "activated_at": "",
            }
        ]
    )

    with pytest.raises(
        ResumenFunnelError,
        match="no son monotónicos",
    ):
        construir_resumen_funnel(
            modelo,
            config.project.as_of_date,
        )


def test_archivo_guardado_es_reproducible(
    tmp_path: Path,
) -> None:
    config, modelo = _generar_modelo()

    resumen = construir_resumen_funnel(
        modelo,
        config.project.as_of_date,
    )

    primera_ruta = guardar_resumen_funnel(
        resumen,
        tmp_path / "primero.csv",
    )

    segunda_ruta = guardar_resumen_funnel(
        resumen,
        tmp_path / "segundo.csv",
    )

    assert (
        primera_ruta.read_bytes()
        == segunda_ruta.read_bytes()
    )

    guardado = pd.read_csv(
        primera_ruta,
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    assert guardado.loc[
        1,
        "previous_stage_count",
    ] == "180"

    assert guardado.loc[
        1,
        "non_progression_from_previous_count",
    ] == "8"