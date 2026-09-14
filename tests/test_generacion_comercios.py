from dataclasses import replace
from pathlib import Path

import pandas as pd

from smb_merchant_lifecycle.generacion_comercios import (
    CANALES_ADQUISICION,
    CIUDADES,
    SEGMENTOS,
    generar_comercios,
    guardar_comercios,
)
from smb_merchant_lifecycle.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.toml"


def test_genera_volumen_y_columnas_esperadas() -> None:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)

    assert len(comercios) == config.project.merchant_count
    assert list(comercios.columns) == [
        "merchant_id",
        "merchant_name",
        "city",
        "business_segment",
        "acquisition_channel",
    ]


def test_claves_y_nombres_son_validos() -> None:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)

    assert comercios["merchant_id"].is_unique
    assert comercios["merchant_id"].str.fullmatch(
        r"M\d{3}"
    ).all()

    assert comercios["merchant_name"].is_unique
    assert (
        comercios["merchant_name"] == "Café Horizonte"
    ).sum() == 1

    assert not comercios.isna().any().any()


def test_dominios_categoricos_son_validos() -> None:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)

    assert set(comercios["city"]).issubset(CIUDADES)
    assert set(
        comercios["business_segment"]
    ).issubset(SEGMENTOS)
    assert set(
        comercios["acquisition_channel"]
    ).issubset(CANALES_ADQUISICION)


def test_misma_semilla_genera_mismo_resultado() -> None:
    config = load_config(CONFIG_PATH)

    primera_generacion = generar_comercios(config)
    segunda_generacion = generar_comercios(config)

    pd.testing.assert_frame_equal(
        primera_generacion,
        segunda_generacion,
    )


def test_semilla_distinta_cambia_el_resultado() -> None:
    config = load_config(CONFIG_PATH)

    config_alternativa = replace(
        config,
        project=replace(
            config.project,
            seed=config.project.seed + 1,
        ),
    )

    primera_generacion = generar_comercios(config)
    segunda_generacion = generar_comercios(
        config_alternativa
    )

    assert not primera_generacion.equals(
        segunda_generacion
    )


def test_guarda_csv_sin_indice(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    comercios = generar_comercios(config)

    ruta = guardar_comercios(
        comercios,
        tmp_path / "merchants.csv",
    )

    datos_guardados = pd.read_csv(
        ruta,
        encoding="utf-8",
    )

    pd.testing.assert_frame_equal(
        datos_guardados,
        comercios,
    )