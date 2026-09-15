"""Generación determinista de eventos de onboarding y manifiesto."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.comercios.generacion import (
    generar_comercios,
    guardar_comercios,
)


ZONA_HORARIA_COLOMBIA = timezone(
    timedelta(hours=-5)
)

SECUENCIA_PRINCIPAL = (
    "REGISTERED",
    "VALIDATION_STARTED",
    "DOCUMENTS_COMPLETED",
    "APPROVED",
    "ACTIVATED",
)

TRANSICIONES_PERMITIDAS: dict[
    str,
    frozenset[str],
] = {
    "REGISTERED": frozenset(
        {
            "VALIDATION_STARTED",
            "ABANDONED",
        }
    ),
    "VALIDATION_STARTED": frozenset(
        {
            "DOCUMENTS_COMPLETED",
            "REJECTED",
            "ABANDONED",
        }
    ),
    "DOCUMENTS_COMPLETED": frozenset(
        {
            "APPROVED",
            "REJECTED",
            "ABANDONED",
        }
    ),
    "APPROVED": frozenset(
        {
            "ACTIVATED",
            "ABANDONED",
        }
    ),
    "ACTIVATED": frozenset(),
    "REJECTED": frozenset(),
    "ABANDONED": frozenset(),
}

RUTAS_ONBOARDING: dict[
    str,
    tuple[str, ...],
] = {
    "ACTIVATED": SECUENCIA_PRINCIPAL,
    "REJECTED_AFTER_VALIDATION": (
        "REGISTERED",
        "VALIDATION_STARTED",
        "REJECTED",
    ),
    "REJECTED_AFTER_DOCUMENTS": (
        "REGISTERED",
        "VALIDATION_STARTED",
        "DOCUMENTS_COMPLETED",
        "REJECTED",
    ),
    "ABANDONED_AFTER_REGISTERED": (
        "REGISTERED",
        "ABANDONED",
    ),
    "ABANDONED_AFTER_VALIDATION": (
        "REGISTERED",
        "VALIDATION_STARTED",
        "ABANDONED",
    ),
    "ABANDONED_AFTER_DOCUMENTS": (
        "REGISTERED",
        "VALIDATION_STARTED",
        "DOCUMENTS_COMPLETED",
        "ABANDONED",
    ),
    "ABANDONED_AFTER_APPROVED": (
        "REGISTERED",
        "VALIDATION_STARTED",
        "DOCUMENTS_COMPLETED",
        "APPROVED",
        "ABANDONED",
    ),
    "IN_PROGRESS_REGISTERED": (
        "REGISTERED",
    ),
    "IN_PROGRESS_VALIDATION": (
        "REGISTERED",
        "VALIDATION_STARTED",
    ),
    "IN_PROGRESS_DOCUMENTS": (
        "REGISTERED",
        "VALIDATION_STARTED",
        "DOCUMENTS_COMPLETED",
    ),
    "IN_PROGRESS_APPROVED": (
        "REGISTERED",
        "VALIDATION_STARTED",
        "DOCUMENTS_COMPLETED",
        "APPROVED",
    ),
}

PROBABILIDADES_RUTA: dict[str, float] = {
    "ACTIVATED": 0.58,
    "REJECTED_AFTER_VALIDATION": 0.06,
    "REJECTED_AFTER_DOCUMENTS": 0.08,
    "ABANDONED_AFTER_REGISTERED": 0.04,
    "ABANDONED_AFTER_VALIDATION": 0.06,
    "ABANDONED_AFTER_DOCUMENTS": 0.06,
    "ABANDONED_AFTER_APPROVED": 0.04,
    "IN_PROGRESS_REGISTERED": 0.02,
    "IN_PROGRESS_VALIDATION": 0.02,
    "IN_PROGRESS_DOCUMENTS": 0.02,
    "IN_PROGRESS_APPROVED": 0.02,
}

RUTAS_ANCLA: dict[
    str,
    tuple[str, ...],
] = {
    "M001": RUTAS_ONBOARDING["ACTIVATED"],
    "M002": RUTAS_ONBOARDING[
        "REJECTED_AFTER_DOCUMENTS"
    ],
    "M003": RUTAS_ONBOARDING[
        "ABANDONED_AFTER_VALIDATION"
    ],
    "M004": RUTAS_ONBOARDING[
        "IN_PROGRESS_DOCUMENTS"
    ],
}

RANGOS_RETRASO_HORAS: dict[
    str,
    tuple[int, int],
] = {
    "VALIDATION_STARTED": (1, 48),
    "DOCUMENTS_COMPLETED": (24, 240),
    "APPROVED": (24, 168),
    "ACTIVATED": (12, 120),
    "REJECTED": (4, 96),
    "ABANDONED": (24, 240),
}


def _crear_generador_aleatorio(
    seed: int,
) -> np.random.Generator:
    """Crea un flujo aleatorio exclusivo para onboarding."""

    secuencia = np.random.SeedSequence(
        [seed, 2]
    )

    return np.random.default_rng(secuencia)


def _seleccionar_ruta(
    merchant_id: str,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Devuelve una ruta ancla o selecciona una sintética."""

    if merchant_id in RUTAS_ANCLA:
        return RUTAS_ANCLA[merchant_id]

    nombres_ruta = tuple(
        PROBABILIDADES_RUTA
    )
    probabilidades = tuple(
        PROBABILIDADES_RUTA.values()
    )

    nombre_elegido = str(
        rng.choice(
            nombres_ruta,
            p=probabilidades,
        )
    )

    return RUTAS_ONBOARDING[nombre_elegido]


def _generar_fecha_registro(
    indice: int,
    config: ProjectConfig,
    rng: np.random.Generator,
) -> datetime:
    """Genera una fecha válida dentro del periodo."""

    total_dias = (
        config.project.end_date
        - config.project.start_date
    ).days

    if indice < len(RUTAS_ANCLA):
        desplazamiento_dias = min(
            indice * 7,
            total_dias,
        )
        hora = 9 + indice
        minuto = 0
    else:
        desplazamiento_dias = int(
            rng.integers(
                0,
                total_dias + 1,
            )
        )
        hora = int(
            rng.integers(8, 18)
        )
        minuto = int(
            rng.choice(
                (0, 15, 30, 45)
            )
        )

    fecha = (
        config.project.start_date
        + timedelta(
            days=desplazamiento_dias
        )
    )

    return datetime.combine(
        fecha,
        time(
            hour=hora,
            minute=minuto,
        ),
        tzinfo=ZONA_HORARIA_COLOMBIA,
    )


def _sumar_retraso(
    evento_destino: str,
    fecha_actual: datetime,
    rng: np.random.Generator,
) -> datetime:
    """Suma un retraso positivo según el evento destino."""

    minimo, maximo = (
        RANGOS_RETRASO_HORAS[
            evento_destino
        ]
    )

    horas = int(
        rng.integers(
            minimo,
            maximo + 1,
        )
    )

    minutos = int(
        rng.choice(
            (0, 15, 30, 45)
        )
    )

    return fecha_actual + timedelta(
        hours=horas,
        minutes=minutos,
    )


def generar_eventos_onboarding(
    comercios: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Genera eventos válidos y ordenados por comercio."""

    if "merchant_id" not in comercios.columns:
        raise ValueError(
            "El catálogo de comercios no contiene "
            "merchant_id."
        )

    if comercios["merchant_id"].duplicated().any():
        raise ValueError(
            "El catálogo contiene merchant_id "
            "duplicados."
        )

    rng = _crear_generador_aleatorio(
        config.project.seed
    )

    fecha_limite = datetime.combine(
        config.project.end_date,
        time(23, 59, 59),
        tzinfo=ZONA_HORARIA_COLOMBIA,
    )

    filas: list[dict[str, str]] = []

    for indice, merchant_id in enumerate(
        comercios["merchant_id"].astype(str)
    ):
        ruta = _seleccionar_ruta(
            merchant_id,
            rng,
        )

        fecha_evento = _generar_fecha_registro(
            indice,
            config,
            rng,
        )

        for posicion, event_type in enumerate(
            ruta
        ):
            if posicion > 0:
                fecha_candidata = _sumar_retraso(
                    event_type,
                    fecha_evento,
                    rng,
                )

                if fecha_candidata > fecha_limite:
                    break

                fecha_evento = fecha_candidata

            filas.append(
                {
                    "onboarding_event_id": (
                        f"OE{len(filas) + 1:04d}"
                    ),
                    "merchant_id": merchant_id,
                    "event_type": event_type,
                    "event_at": (
                        fecha_evento.isoformat(
                            timespec="seconds"
                        )
                    ),
                }
            )

    return pd.DataFrame(
        filas,
        columns=[
            "onboarding_event_id",
            "merchant_id",
            "event_type",
            "event_at",
        ],
    )


def guardar_eventos_onboarding(
    eventos: pd.DataFrame,
    ruta: str | Path = Path(
        "data/raw/onboarding_events.csv"
    ),
) -> Path:
    """Guarda los eventos con formato reproducible."""

    ruta_salida = Path(ruta)

    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    eventos.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return ruta_salida


def _calcular_sha256(
    ruta: Path,
) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    return hashlib.sha256(
        ruta.read_bytes()
    ).hexdigest()


def _contar_ultimos_eventos(
    eventos: pd.DataFrame,
) -> dict[str, int]:
    """Cuenta el último evento observado por comercio."""

    ultimos = (
        eventos.sort_values(
            [
                "merchant_id",
                "event_at",
                "onboarding_event_id",
            ]
        )
        .groupby(
            "merchant_id",
            sort=True,
        )
        .tail(1)
    )

    return {
        str(evento): int(conteo)
        for evento, conteo in (
            ultimos["event_type"]
            .value_counts()
            .sort_index()
            .items()
        )
    }


def guardar_datos_fase2(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    config: ProjectConfig,
    directorio: str | Path = Path(
        "data/raw"
    ),
) -> dict[str, object]:
    """Guarda los CSV y un manifiesto determinista."""

    directorio_salida = Path(
        directorio
    )

    directorio_salida.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_comercios = guardar_comercios(
        comercios,
        directorio_salida / "merchants.csv",
    )

    ruta_eventos = guardar_eventos_onboarding(
        eventos,
        directorio_salida
        / "onboarding_events.csv",
    )

    conteos_evento = {
        str(evento): int(conteo)
        for evento, conteo in (
            eventos["event_type"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    manifiesto: dict[str, object] = {
        "schema_version": "1.0",
        "seed": config.project.seed,
        "period": {
            "start_date": (
                config.project.start_date.isoformat()
            ),
            "end_date": (
                config.project.end_date.isoformat()
            ),
            "as_of_date": (
                config.project.as_of_date.isoformat()
            ),
        },
        "timezone": "UTC-05:00",
        "configured_merchant_count": (
            config.project.merchant_count
        ),
        "outputs": {
            "merchants.csv": {
                "rows": int(len(comercios)),
                "sha256": _calcular_sha256(
                    ruta_comercios
                ),
            },
            "onboarding_events.csv": {
                "rows": int(len(eventos)),
                "sha256": _calcular_sha256(
                    ruta_eventos
                ),
            },
        },
        "onboarding": {
            "event_counts": conteos_evento,
            "last_event_counts": (
                _contar_ultimos_eventos(
                    eventos
                )
            ),
        },
        "synthetic_assumptions": {
            "route_probabilities": (
                PROBABILIDADES_RUTA
            ),
            "anchor_routes": {
                merchant_id: list(ruta)
                for merchant_id, ruta
                in RUTAS_ANCLA.items()
            },
        },
    }

    ruta_manifiesto = (
        directorio_salida
        / "generation_manifest.json"
    )

    ruta_manifiesto.write_text(
        json.dumps(
            manifiesto,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return manifiesto


def main() -> None:
    """Genera los archivos raw y el manifiesto."""

    config = load_config()
    comercios = generar_comercios(config)

    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )

    guardar_datos_fase2(
        comercios,
        eventos,
        config,
    )

    print(
        "Generación completada: "
        f"{len(comercios)} comercios y "
        f"{len(eventos)} eventos."
    )


if __name__ == "__main__":
    main()