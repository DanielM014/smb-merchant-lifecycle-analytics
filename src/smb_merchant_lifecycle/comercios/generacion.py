"""Generación determinista del catálogo maestro de comercios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smb_merchant_lifecycle.config import ProjectConfig, load_config


NOMBRE_ANCLA = "Café Horizonte"

TIPOS_NEGOCIO = (
    "Café",
    "Panadería",
    "Restaurante",
    "Mercado",
    "Ferretería",
    "Farmacia",
    "Papelería",
    "Librería",
    "Taller",
    "Boutique",
    "Distribuidora",
    "Miscelánea",
    "Heladería",
    "Floristería",
    "Óptica",
)

DISTINTIVOS = (
    "Horizonte",
    "Aurora",
    "Roble",
    "Nébula",
    "Sol",
    "Cedro",
    "Marea",
    "Prisma",
    "Boreal",
    "Coral",
    "Guayacán",
    "Manzano",
    "Litoral",
    "Andino",
    "Esmeralda",
    "Colibrí",
    "Sabana",
    "Orquídea",
    "Nogal",
    "Granada",
    "Alameda",
)

CIUDADES = (
    "Bogotá",
    "Medellín",
    "Cali",
    "Barranquilla",
    "Bucaramanga",
    "Cartagena",
    "Pereira",
)
PROBABILIDADES_CIUDAD = (
    0.32,
    0.20,
    0.17,
    0.11,
    0.08,
    0.07,
    0.05,
)

SEGMENTOS = (
    "SMALL",
    "MEDIUM",
)
PROBABILIDADES_SEGMENTO = (
    0.78,
    0.22,
)

CANALES_ADQUISICION = (
    "DIGITAL",
    "VENTA_DIRECTA",
    "ALIADOS",
    "REFERIDOS",
)
PROBABILIDADES_CANAL = (
    0.35,
    0.30,
    0.20,
    0.15,
)


def _crear_generador_aleatorio(seed: int) -> np.random.Generator:
    """Crea un flujo aleatorio estable y exclusivo para comercios."""

    secuencia = np.random.SeedSequence([seed, 1])
    return np.random.default_rng(secuencia)


def _generar_nombres(
    cantidad: int,
    rng: np.random.Generator,
) -> list[str]:
    """Genera nombres legibles sin usarlos como claves del modelo."""

    candidatos = [
        f"{tipo} {distintivo}"
        for tipo in TIPOS_NEGOCIO
        for distintivo in DISTINTIVOS
        if f"{tipo} {distintivo}" != NOMBRE_ANCLA
    ]

    cantidad_adicional = cantidad - 1

    if cantidad_adicional > len(candidatos):
        maximo = len(candidatos) + 1
        raise ValueError(
            "No existen suficientes combinaciones de nombres para generar "
            f"{cantidad} comercios únicos. Máximo admitido: {maximo}."
        )

    indices = rng.choice(
        len(candidatos),
        size=cantidad_adicional,
        replace=False,
    )

    nombres_aleatorios = [
        candidatos[int(indice)]
        for indice in indices
    ]

    return [NOMBRE_ANCLA, *nombres_aleatorios]


def generar_comercios(config: ProjectConfig) -> pd.DataFrame:
    """Genera una fila maestra por comercio sintético."""

    cantidad = config.project.merchant_count
    rng = _crear_generador_aleatorio(config.project.seed)
    ancho_id = max(3, len(str(cantidad)))

    comercios = pd.DataFrame(
        {
            "merchant_id": [
                f"M{numero:0{ancho_id}d}"
                for numero in range(1, cantidad + 1)
            ],
            "merchant_name": _generar_nombres(
                cantidad,
                rng,
            ),
            "city": rng.choice(
                CIUDADES,
                size=cantidad,
                p=PROBABILIDADES_CIUDAD,
            ),
            "business_segment": rng.choice(
                SEGMENTOS,
                size=cantidad,
                p=PROBABILIDADES_SEGMENTO,
            ),
            "acquisition_channel": rng.choice(
                CANALES_ADQUISICION,
                size=cantidad,
                p=PROBABILIDADES_CANAL,
            ),
        }
    )

    return comercios


def guardar_comercios(
    comercios: pd.DataFrame,
    ruta: str | Path = Path("data/raw/merchants.csv"),
) -> Path:
    """Guarda el catálogo con codificación y saltos reproducibles."""

    ruta_salida = Path(ruta)
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comercios.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return ruta_salida


def main() -> None:
    """Genera el primer archivo raw del proyecto."""

    config = load_config()
    comercios = generar_comercios(config)
    ruta = guardar_comercios(comercios)

    print(
        f"Generados {len(comercios)} comercios en {ruta}."
    )


if __name__ == "__main__":
    main()