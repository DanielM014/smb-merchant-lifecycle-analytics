from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.contratos_datos import (
    ContratoEstructuralError,
    validar_contrato_comercios,
    validar_contrato_eventos_onboarding,
)
from smb_merchant_lifecycle.generacion_comercios import (
    generar_comercios,
)
from smb_merchant_lifecycle.generacion_onboarding import (
    generar_eventos_onboarding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def test_datos_generados_cumplen_contratos() -> None:
    config, comercios, eventos = _generar_datos()

    comercios_validados = validar_contrato_comercios(
        comercios
    )
    eventos_validados = (
        validar_contrato_eventos_onboarding(
            eventos,
            config,
        )
    )

    pd.testing.assert_frame_equal(
        comercios_validados,
        comercios,
    )
    pd.testing.assert_frame_equal(
        eventos_validados,
        eventos,
    )


def test_archivos_raw_publicados_cumplen_contratos() -> None:
    config = load_config(CONFIG_PATH)

    comercios = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "merchants.csv",
        encoding="utf-8",
    )
    eventos = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "raw"
        / "onboarding_events.csv",
        encoding="utf-8",
    )

    validar_contrato_comercios(
        comercios
    )
    validar_contrato_eventos_onboarding(
        eventos,
        config,
    )


def test_nombre_comercial_repetido_no_rompe_contrato() -> None:
    _, comercios, _ = _generar_datos()
    comercios = comercios.copy()

    comercios.loc[
        1,
        "merchant_name",
    ] = comercios.loc[
        0,
        "merchant_name",
    ]

    validar_contrato_comercios(comercios)


def test_columna_faltante_detiene_validacion() -> None:
    _, comercios, _ = _generar_datos()

    sin_ciudad = comercios.drop(
        columns="city"
    )

    with pytest.raises(
        ContratoEstructuralError,
        match="faltantes=city",
    ):
        validar_contrato_comercios(
            sin_ciudad
        )


def test_columnas_extra_o_desordenadas_detienen_validacion() -> None:
    _, comercios, _ = _generar_datos()

    con_extra = comercios.assign(
        estado_manual="ACTIVATED"
    )

    with pytest.raises(
        ContratoEstructuralError,
        match="adicionales=estado_manual",
    ):
        validar_contrato_comercios(
            con_extra
        )

    desordenadas = comercios[
        list(reversed(comercios.columns))
    ]

    with pytest.raises(
        ContratoEstructuralError,
        match="orden de columnas inválido",
    ):
        validar_contrato_comercios(
            desordenadas
        )

    repetidas = pd.concat(
        [
            comercios,
            comercios[["merchant_id"]],
        ],
        axis="columns",
    )

    with pytest.raises(
        ContratoEstructuralError,
        match="repetidas=merchant_id",
    ):
        validar_contrato_comercios(
            repetidas
        )


def test_merchant_id_duplicado_es_invalido() -> None:
    _, comercios, _ = _generar_datos()
    comercios = comercios.copy()

    comercios.loc[
        1,
        "merchant_id",
    ] = comercios.loc[
        0,
        "merchant_id",
    ]

    with pytest.raises(
        pa.errors.SchemaErrors
    ):
        validar_contrato_comercios(
            comercios
        )


def test_vacios_y_dominios_de_comercio_son_invalidos() -> None:
    _, comercios, _ = _generar_datos()
    comercios = comercios.copy()

    comercios.loc[0, "merchant_name"] = "   "
    comercios.loc[1, "city"] = "Ciudad inexistente"

    with pytest.raises(
        pa.errors.SchemaErrors
    ):
        validar_contrato_comercios(
            comercios
        )


def test_clave_y_dominio_de_evento_son_invalidos() -> None:
    config, _, eventos = _generar_datos()
    eventos = eventos.copy()

    eventos.loc[
        1,
        "onboarding_event_id",
    ] = eventos.loc[
        0,
        "onboarding_event_id",
    ]
    eventos.loc[
        2,
        "event_type",
    ] = "FIRST_TRANSACTION"

    with pytest.raises(
        pa.errors.SchemaErrors
    ):
        validar_contrato_eventos_onboarding(
            eventos,
            config,
        )


def test_event_at_exige_periodo_y_zona_horaria() -> None:
    config, _, eventos = _generar_datos()
    eventos = eventos.copy()

    eventos.loc[
        0,
        "event_at",
    ] = "2025-01-01T09:00:00"
    eventos.loc[
        1,
        "event_at",
    ] = "2026-09-01T09:00:00-05:00"

    with pytest.raises(
        pa.errors.SchemaErrors
    ):
        validar_contrato_eventos_onboarding(
            eventos,
            config,
        )