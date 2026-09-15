from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

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
from smb_merchant_lifecycle.onboarding.validacion import (
    guardar_resultado_validacion,
    validar_integridad_onboarding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


def _generar_datos() -> tuple[
    ProjectConfig,
    pd.DataFrame,
    pd.DataFrame,
]:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)
    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )

    return config, comercios, eventos


def _agregar_evento(
    eventos: pd.DataFrame,
    *,
    event_id: str,
    merchant_id: str,
    event_type: str,
    event_at: datetime,
) -> pd.DataFrame:
    nuevo = pd.DataFrame(
        [
            {
                "onboarding_event_id": event_id,
                "merchant_id": merchant_id,
                "event_type": event_type,
                "event_at": event_at.isoformat(
                    timespec="seconds"
                ),
            }
        ],
        columns=eventos.columns,
    )

    return pd.concat(
        [eventos, nuevo],
        ignore_index=True,
    )


def _reglas(
    resultado: object,
) -> set[str]:
    return set(
        resultado.violaciones["rule_id"]
    )


def test_datos_oficiales_pasan_sin_rechazos() -> None:
    config, comercios, eventos = _generar_datos()

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    assert len(resultado.comercios_validos) == 180
    assert len(resultado.eventos_validos) == 756
    assert resultado.violaciones.empty
    assert resultado.registros_rechazados.empty


def test_evento_huerfano_se_rechaza() -> None:
    config, comercios, eventos = _generar_datos()

    eventos = _agregar_evento(
        eventos,
        event_id="OE9999",
        merchant_id="M999",
        event_type="REGISTERED",
        event_at=datetime.fromisoformat(
            "2025-01-01T08:00:00-05:00"
        ),
    )

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    assert (
        "ONBOARDING_MERCHANT_NOT_FOUND_OR_INVALID"
        in _reglas(resultado)
    )
    assert len(resultado.comercios_validos) == 180
    assert len(resultado.eventos_validos) == 756
    assert len(resultado.registros_rechazados) == 1


def test_ruta_sin_registered_entra_en_cuarentena() -> None:
    config, comercios, eventos = _generar_datos()

    mascara = (
        (eventos["merchant_id"] == "M004")
        & (eventos["event_type"] == "REGISTERED")
    )
    eventos = eventos.loc[~mascara].reset_index(
        drop=True
    )

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    assert (
        "MERCHANT_REGISTERED_COUNT_INVALID"
        in _reglas(resultado)
    )
    assert "M004" not in set(
        resultado.comercios_validos["merchant_id"]
    )
    assert "M004" not in set(
        resultado.eventos_validos["merchant_id"]
    )


def test_etapa_repetida_invalida_ruta_completa() -> None:
    config, comercios, eventos = _generar_datos()

    ruta = eventos.loc[
        eventos["merchant_id"] == "M001"
    ]
    inicio = datetime.fromisoformat(
        ruta.loc[
            ruta["event_type"]
            == "VALIDATION_STARTED",
            "event_at",
        ].iloc[0]
    )
    fin = datetime.fromisoformat(
        ruta.loc[
            ruta["event_type"]
            == "DOCUMENTS_COMPLETED",
            "event_at",
        ].iloc[0]
    )

    eventos = _agregar_evento(
        eventos,
        event_id="OE9998",
        merchant_id="M001",
        event_type="VALIDATION_STARTED",
        event_at=inicio + (fin - inicio) / 2,
    )

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    assert (
        "ONBOARDING_STAGE_REPEATED"
        in _reglas(resultado)
    )
    assert "M001" not in set(
        resultado.comercios_validos["merchant_id"]
    )
    assert "M001" not in set(
        resultado.eventos_validos["merchant_id"]
    )


def test_transicion_imposible_invalida_ruta() -> None:
    config, comercios, eventos = _generar_datos()
    eventos = eventos.copy()

    mascara = (
        (eventos["merchant_id"] == "M004")
        & (
            eventos["event_type"]
            == "VALIDATION_STARTED"
        )
    )
    eventos.loc[mascara, "event_type"] = "APPROVED"

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    assert (
        "ONBOARDING_TRANSITION_INVALID"
        in _reglas(resultado)
    )
    assert "M004" not in set(
        resultado.eventos_validos["merchant_id"]
    )


def test_evento_antes_de_registered_invalida_ruta() -> None:
    config, comercios, eventos = _generar_datos()
    eventos = eventos.copy()

    mascara_registered = (
        (eventos["merchant_id"] == "M004")
        & (eventos["event_type"] == "REGISTERED")
    )
    fecha_registered = datetime.fromisoformat(
        eventos.loc[
            mascara_registered,
            "event_at",
        ].iloc[0]
    )
    mascara_validacion = (
        (eventos["merchant_id"] == "M004")
        & (
            eventos["event_type"]
            == "VALIDATION_STARTED"
        )
    )
    eventos.loc[
        mascara_validacion,
        "event_at",
    ] = (
        fecha_registered
        - timedelta(hours=1)
    ).isoformat(timespec="seconds")

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    assert (
        "ONBOARDING_EVENT_BEFORE_REGISTERED"
        in _reglas(resultado)
    )
    assert "M004" not in set(
        resultado.eventos_validos["merchant_id"]
    )


def test_evento_despues_de_terminal_invalida_ruta() -> None:
    config, comercios, eventos = _generar_datos()

    fecha_terminal = datetime.fromisoformat(
        eventos.loc[
            (eventos["merchant_id"] == "M003")
            & (eventos["event_type"] == "ABANDONED"),
            "event_at",
        ].iloc[0]
    )
    eventos = _agregar_evento(
        eventos,
        event_id="OE9997",
        merchant_id="M003",
        event_type="APPROVED",
        event_at=fecha_terminal + timedelta(hours=1),
    )

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    assert (
        "ONBOARDING_EVENT_AFTER_TERMINAL"
        in _reglas(resultado)
    )
    assert "M003" not in set(
        resultado.eventos_validos["merchant_id"]
    )


def test_event_type_fuera_de_contrato_genera_regla_estable() -> None:
    config, comercios, eventos = _generar_datos()
    eventos = eventos.copy()

    mascara = (
        (eventos["merchant_id"] == "M004")
        & (
            eventos["event_type"]
            == "DOCUMENTS_COMPLETED"
        )
    )
    eventos.loc[
        mascara,
        "event_type",
    ] = "FIRST_TRANSACTION"

    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    assert (
        "ONBOARDING_EVENT_TYPE_INVALID"
        in _reglas(resultado)
    )
    assert "M004" not in set(
        resultado.comercios_validos["merchant_id"]
    )
    assert "M004" not in set(
        resultado.eventos_validos["merchant_id"]
    )


def test_salida_guardada_concilia_validos_y_rechazados(
    tmp_path: Path,
) -> None:
    config, comercios, eventos = _generar_datos()

    eventos = _agregar_evento(
        eventos,
        event_id="OE9996",
        merchant_id="M999",
        event_type="REGISTERED",
        event_at=datetime.fromisoformat(
            "2025-01-01T08:00:00-05:00"
        ),
    )
    resultado = validar_integridad_onboarding(
        comercios,
        eventos,
        config,
    )

    rutas = guardar_resultado_validacion(
        resultado,
        tmp_path / "procesada",
        tmp_path / "rechazada",
    )

    assert set(rutas) == {
        "merchants",
        "onboarding_events",
        "quality_violations",
        "rejected_records",
    }
    assert all(
        ruta.exists()
        for ruta in rutas.values()
    )

    comercios_guardados = pd.read_csv(
        rutas["merchants"],
        encoding="utf-8",
    )
    eventos_guardados = pd.read_csv(
        rutas["onboarding_events"],
        encoding="utf-8",
    )
    violaciones_guardadas = pd.read_csv(
        rutas["quality_violations"],
        encoding="utf-8",
    )
    rechazados_guardados = pd.read_csv(
        rutas["rejected_records"],
        encoding="utf-8",
    )

    assert len(comercios_guardados) == 180
    assert len(eventos_guardados) == 756
    assert len(violaciones_guardadas) == 1
    assert len(rechazados_guardados) == 1

    assert (
        len(comercios)
        + len(eventos)
        == len(comercios_guardados)
        + len(eventos_guardados)
        + len(rechazados_guardados)
    )
    assert (
        len(rechazados_guardados)
        <= len(violaciones_guardadas)
    )
