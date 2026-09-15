import hashlib
import json
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from smb_merchant_lifecycle.config import load_config
from smb_merchant_lifecycle.comercios.generacion import (
    generar_comercios,
)
from smb_merchant_lifecycle.onboarding.generacion import (
    PROBABILIDADES_RUTA,
    RUTAS_ANCLA,
    TRANSICIONES_PERMITIDAS,
    generar_eventos_onboarding,
    guardar_datos_fase2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


def _generar_datos() -> tuple[
    object,
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


def _calcular_hash(
    ruta: Path,
) -> str:
    return hashlib.sha256(
        ruta.read_bytes()
    ).hexdigest()


def test_esquema_claves_e_integridad_referencial() -> None:
    _, comercios, eventos = _generar_datos()

    assert list(eventos.columns) == [
        "onboarding_event_id",
        "merchant_id",
        "event_type",
        "event_at",
    ]

    assert not eventos.isna().any().any()

    assert eventos[
        "onboarding_event_id"
    ].is_unique

    assert eventos[
        "onboarding_event_id"
    ].str.fullmatch(
        r"OE\d{4}"
    ).all()

    assert set(
        eventos["merchant_id"]
    ) == set(
        comercios["merchant_id"]
    )


def test_cada_comercio_tiene_un_registered() -> None:
    _, comercios, eventos = _generar_datos()

    registrados = (
        eventos.loc[
            eventos["event_type"]
            == "REGISTERED"
        ]
        .groupby("merchant_id")
        .size()
    )

    assert len(registrados) == len(comercios)
    assert (registrados == 1).all()


def test_secuencias_y_fechas_son_validas() -> None:
    _, _, eventos = _generar_datos()

    for _, grupo in eventos.groupby(
        "merchant_id",
        sort=False,
    ):
        ordenado = grupo.sort_values(
            [
                "event_at",
                "onboarding_event_id",
            ]
        )

        tipos = ordenado[
            "event_type"
        ].tolist()

        fechas = [
            datetime.fromisoformat(valor)
            for valor in ordenado["event_at"]
        ]

        assert tipos[0] == "REGISTERED"
        assert len(tipos) == len(set(tipos))

        assert all(
            siguiente
            in TRANSICIONES_PERMITIDAS[actual]
            for actual, siguiente
            in pairwise(tipos)
        )

        assert all(
            anterior < siguiente
            for anterior, siguiente
            in pairwise(fechas)
        )


def test_rutas_ancla_son_reproducibles() -> None:
    _, _, eventos = _generar_datos()

    for merchant_id, ruta_esperada in (
        RUTAS_ANCLA.items()
    ):
        ruta_observada = (
            eventos.loc[
                eventos["merchant_id"]
                == merchant_id
            ]
            .sort_values(
                [
                    "event_at",
                    "onboarding_event_id",
                ]
            )["event_type"]
            .tolist()
        )

        assert ruta_observada == list(
            ruta_esperada
        )


def test_dominio_periodo_y_zona_horaria() -> None:
    config, _, eventos = _generar_datos()

    assert set(
        eventos["event_type"]
    ).issubset(
        TRANSICIONES_PERMITIDAS
    )

    assert (
        "FIRST_TRANSACTION"
        not in set(eventos["event_type"])
    )

    fechas = [
        datetime.fromisoformat(valor)
        for valor in eventos["event_at"]
    ]

    assert min(
        fecha.date()
        for fecha in fechas
    ) >= config.project.start_date

    assert max(
        fecha.date()
        for fecha in fechas
    ) <= config.project.end_date

    assert {
        fecha.utcoffset()
        for fecha in fechas
    } == {
        timedelta(hours=-5)
    }

    assert sum(
        PROBABILIDADES_RUTA.values()
    ) == pytest.approx(1.0)


def test_misma_semilla_genera_mismos_eventos() -> None:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)

    primera_generacion = (
        generar_eventos_onboarding(
            comercios,
            config,
        )
    )

    segunda_generacion = (
        generar_eventos_onboarding(
            comercios,
            config,
        )
    )

    pd.testing.assert_frame_equal(
        primera_generacion,
        segunda_generacion,
    )


def test_manifiesto_concilia_archivos_y_hashes(
    tmp_path: Path,
) -> None:
    config, comercios, eventos = (
        _generar_datos()
    )

    primera_salida = tmp_path / "primera"
    segunda_salida = tmp_path / "segunda"

    primer_manifiesto = guardar_datos_fase2(
        comercios,
        eventos,
        config,
        primera_salida,
    )

    segundo_manifiesto = guardar_datos_fase2(
        comercios,
        eventos,
        config,
        segunda_salida,
    )

    assert primer_manifiesto == segundo_manifiesto

    manifiesto_guardado = json.loads(
        (
            primera_salida
            / "generation_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifiesto_guardado
        == primer_manifiesto
    )

    salidas = primer_manifiesto["outputs"]

    assert (
        salidas["merchants.csv"]["rows"]
        == len(comercios)
    )

    assert (
        salidas[
            "onboarding_events.csv"
        ]["rows"]
        == len(eventos)
    )

    assert (
        salidas["merchants.csv"]["sha256"]
        == _calcular_hash(
            primera_salida
            / "merchants.csv"
        )
    )

    assert (
        salidas[
            "onboarding_events.csv"
        ]["sha256"]
        == _calcular_hash(
            primera_salida
            / "onboarding_events.csv"
        )
    )

    conteos = primer_manifiesto[
        "onboarding"
    ]

    assert sum(
        conteos["event_counts"].values()
    ) == len(eventos)

    assert sum(
        conteos["last_event_counts"].values()
    ) == len(comercios)

    assert (
        (
            primera_salida
            / "generation_manifest.json"
        ).read_bytes()
        == (
            segunda_salida
            / "generation_manifest.json"
        ).read_bytes()
    )