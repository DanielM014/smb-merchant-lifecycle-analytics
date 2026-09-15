from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.comercios.generacion import (
    generar_comercios,
)
from smb_merchant_lifecycle.onboarding.generacion import (
    generar_eventos_onboarding,
)
from smb_merchant_lifecycle.onboarding.modelo import (
    COLUMNAS_MODELO_ONBOARDING,
    ModeloOnboardingError,
    construir_modelo_onboarding,
    guardar_modelo_onboarding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


def _generar_modelo() -> tuple[
    ProjectConfig,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)
    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )
    modelo = construir_modelo_onboarding(
        comercios,
        eventos,
        config,
    )

    return config, comercios, eventos, modelo


def _fila_comercio(
    modelo: pd.DataFrame,
    merchant_id: str,
) -> pd.Series:
    return modelo.loc[
        modelo["merchant_id"] == merchant_id
    ].iloc[0]


def test_modelo_tiene_una_fila_por_comercio() -> None:
    _, comercios, _, modelo = _generar_modelo()

    assert list(modelo.columns) == list(
        COLUMNAS_MODELO_ONBOARDING
    )
    assert len(modelo) == len(comercios) == 180
    assert modelo["merchant_id"].is_unique
    assert set(modelo["merchant_id"]) == set(
        comercios["merchant_id"]
    )


def test_rutas_ancla_producen_estados_correctos() -> None:
    _, _, _, modelo = _generar_modelo()

    cafe_horizonte = _fila_comercio(
        modelo,
        "M001",
    )
    rechazado = _fila_comercio(
        modelo,
        "M002",
    )
    abandonado = _fila_comercio(
        modelo,
        "M003",
    )
    en_progreso = _fila_comercio(
        modelo,
        "M004",
    )

    assert (
        cafe_horizonte["onboarding_status"]
        == "ACTIVATED"
    )
    assert (
        cafe_horizonte["highest_stage_reached"]
        == "ACTIVATED"
    )
    assert bool(
        cafe_horizonte["is_technically_activated"]
    )

    assert rechazado["onboarding_status"] == "REJECTED"
    assert (
        rechazado["highest_stage_reached"]
        == "DOCUMENTS_COMPLETED"
    )
    assert (
        rechazado["exit_from_stage"]
        == "DOCUMENTS_COMPLETED"
    )

    assert abandonado["onboarding_status"] == "ABANDONED"
    assert (
        abandonado["highest_stage_reached"]
        == "VALIDATION_STARTED"
    )
    assert (
        abandonado["exit_from_stage"]
        == "VALIDATION_STARTED"
    )

    assert (
        en_progreso["onboarding_status"]
        == "IN_PROGRESS"
    )
    assert (
        en_progreso["highest_stage_reached"]
        == "DOCUMENTS_COMPLETED"
    )
    assert (
        en_progreso["last_onboarding_event_type"]
        == "DOCUMENTS_COMPLETED"
    )


def test_conteos_del_modelo_concilian_con_eventos() -> None:
    _, _, _, modelo = _generar_modelo()

    assert modelo["onboarding_status"].value_counts().to_dict() == {
        "ACTIVATED": 98,
        "ABANDONED": 34,
        "REJECTED": 25,
        "IN_PROGRESS": 23,
    }

    campos_etapa = {
        "registered_at": 180,
        "validation_started_at": 172,
        "documents_completed_at": 141,
        "approved_at": 106,
        "activated_at": 98,
    }

    for campo, conteo_esperado in campos_etapa.items():
        assert int(
            modelo[campo].ne("").sum()
        ) == conteo_esperado

    assert int(
        modelo[
            "is_commercial_activation_30d_eligible"
        ].sum()
    ) == 168
    assert int(
        modelo["is_technically_activated"].sum()
    ) == 98


def test_relojes_temporales_se_calculan_desde_su_origen() -> None:
    config, _, _, modelo = _generar_modelo()

    for _, fila in modelo.iterrows():
        registered_date = datetime.fromisoformat(
            fila["registered_at"]
        ).date()
        last_event_date = datetime.fromisoformat(
            fila["last_onboarding_event_at"]
        ).date()

        assert fila["days_since_registered"] == (
            config.project.as_of_date
            - registered_date
        ).days
        assert fila["days_in_current_stage"] == (
            config.project.as_of_date
            - last_event_date
        ).days
        assert bool(
            fila[
                "is_commercial_activation_30d_eligible"
            ]
        ) == (
            fila["days_since_registered"]
            >= config.onboarding.first_transaction_window_days
        )

        if fila["activated_at"]:
            activated_date = datetime.fromisoformat(
                fila["activated_at"]
            ).date()
            assert fila[
                "days_to_technical_activation"
            ] == (
                activated_date
                - registered_date
            ).days
        else:
            assert pd.isna(
                fila["days_to_technical_activation"]
            )


def test_exit_from_stage_solo_aplica_a_salidas() -> None:
    _, _, _, modelo = _generar_modelo()

    con_salida = modelo["onboarding_status"].isin(
        ["REJECTED", "ABANDONED"]
    )

    assert modelo.loc[
        con_salida,
        "exit_from_stage",
    ].ne("").all()
    assert modelo.loc[
        ~con_salida,
        "exit_from_stage",
    ].eq("").all()


def test_orden_de_entrada_no_cambia_el_modelo() -> None:
    config, comercios, eventos, esperado = (
        _generar_modelo()
    )

    comercios_desordenados = comercios.sample(
        frac=1,
        random_state=11,
    ).reset_index(drop=True)
    eventos_desordenados = eventos.sample(
        frac=1,
        random_state=12,
    ).reset_index(drop=True)

    observado = construir_modelo_onboarding(
        comercios_desordenados,
        eventos_desordenados,
        config,
    )

    pd.testing.assert_frame_equal(
        observado,
        esperado,
    )


def test_modelo_rechaza_entrada_con_ruta_invalida() -> None:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)
    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )
    eventos = eventos.copy()

    mascara = (
        (eventos["merchant_id"] == "M004")
        & (
            eventos["event_type"]
            == "VALIDATION_STARTED"
        )
    )
    eventos.loc[mascara, "event_type"] = "APPROVED"

    with pytest.raises(
        ModeloOnboardingError,
        match="ONBOARDING_TRANSITION_INVALID",
    ):
        construir_modelo_onboarding(
            comercios,
            eventos,
            config,
        )


def test_construccion_no_modifica_las_entradas() -> None:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)
    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )
    comercios_originales = comercios.copy(
        deep=True
    )
    eventos_originales = eventos.copy(
        deep=True
    )

    construir_modelo_onboarding(
        comercios,
        eventos,
        config,
    )

    pd.testing.assert_frame_equal(
        comercios,
        comercios_originales,
    )
    pd.testing.assert_frame_equal(
        eventos,
        eventos_originales,
    )


def test_archivo_guardado_es_reproducible(
    tmp_path: Path,
) -> None:
    _, _, _, modelo = _generar_modelo()

    primera_ruta = guardar_modelo_onboarding(
        modelo,
        tmp_path / "primera.csv",
    )
    segunda_ruta = guardar_modelo_onboarding(
        modelo,
        tmp_path / "segunda.csv",
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

    assert list(guardado.columns) == list(
        COLUMNAS_MODELO_ONBOARDING
    )
    assert len(guardado) == 180
    assert guardado["merchant_id"].is_unique
