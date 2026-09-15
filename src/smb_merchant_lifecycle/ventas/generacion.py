"""Generación determinista de actividad comercial diaria sintética."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)


COLUMNAS_VENTAS_DIARIAS = (
    "merchant_id",
    "sales_date",
    "approved_transactions",
    "sales_amount_cop",
)

ESCENARIOS_TENDENCIA = (
    "STABLE_28D",
    "DECLINE_28D",
    "GROWTH_28D",
    "ZERO_CURRENT_28D",
)

PROBABILIDADES_TENDENCIA = (
    0.55,
    0.20,
    0.20,
    0.05,
)

ANCLAS_PRIMERA_TRANSACCION: dict[
    str,
    tuple[str, int | None],
] = {
    "M001": ("ACTIVATED", 2),
    "M006": ("ACTIVATED", 2),
    "M007": ("ACTIVATED", 2),
    "M008": ("ACTIVATED", 2),
    "M009": ("REGISTERED", 32),
    "M010": ("NONE", None),
}

ANCLAS_VENTANAS_28D: dict[
    str,
    tuple[int, int],
] = {
    "M001": (50_000_000, 36_000_000),
    "M006": (30_000_000, 42_000_000),
    "M007": (28_000_000, 0),
    "M008": (40_000_000, 42_000_000),
}


class GeneracionVentasError(ValueError):
    """Indica que las entradas no permiten generar ventas seguras."""


def _crear_generador_aleatorio(
    seed: int,
) -> np.random.Generator:
    """Crea un flujo independiente para generar ventas."""

    secuencia = np.random.SeedSequence(
        [seed, 3]
    )

    return np.random.default_rng(
        secuencia
    )


def _fecha_evento(valor: str) -> date:
    """Extrae la fecha calendario de un timestamp ISO validado."""

    try:
        return datetime.fromisoformat(
            valor
        ).date()
    except (TypeError, ValueError) as exc:
        raise GeneracionVentasError(
            "Timestamp de onboarding inválido: "
            f"{valor!r}."
        ) from exc


def _validar_entradas(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    config: ProjectConfig,
) -> None:
    """Valida el grano y las columnas de las fuentes."""

    columnas_comercios = {
        "merchant_id",
        "business_segment",
    }
    columnas_eventos = {
        "merchant_id",
        "event_type",
        "event_at",
    }

    faltantes_comercios = sorted(
        columnas_comercios
        - set(comercios.columns)
    )
    faltantes_eventos = sorted(
        columnas_eventos
        - set(eventos.columns)
    )

    if faltantes_comercios:
        raise GeneracionVentasError(
            "Faltan columnas de comercios: "
            f"{', '.join(faltantes_comercios)}."
        )

    if faltantes_eventos:
        raise GeneracionVentasError(
            "Faltan columnas de onboarding: "
            f"{', '.join(faltantes_eventos)}."
        )

    if comercios[
        "merchant_id"
    ].duplicated().any():
        raise GeneracionVentasError(
            "El catálogo contiene merchant_id "
            "duplicados."
        )

    segmentos_invalidos = sorted(
        set(
            comercios["business_segment"]
        )
        - {"SMALL", "MEDIUM"}
    )

    if segmentos_invalidos:
        raise GeneracionVentasError(
            "Existen business_segment inválidos: "
            f"{', '.join(segmentos_invalidos)}."
        )

    if config.project.end_date != (
        config.project.as_of_date
        - timedelta(days=1)
    ):
        raise GeneracionVentasError(
            "end_date debe ser el día inmediatamente "
            "anterior a as_of_date para generar "
            "cobertura diaria completa."
        )


def _extraer_fechas_onboarding(
    eventos: pd.DataFrame,
) -> dict[str, tuple[date, date]]:
    """Obtiene REGISTERED y ACTIVATED para los activados."""

    relevantes = eventos.loc[
        eventos["event_type"].isin(
            [
                "REGISTERED",
                "ACTIVATED",
            ]
        )
    ].copy()

    duplicados = relevantes.duplicated(
        subset=[
            "merchant_id",
            "event_type",
        ],
        keep=False,
    )

    if duplicados.any():
        raise GeneracionVentasError(
            "REGISTERED y ACTIVATED no pueden "
            "repetirse por comercio."
        )

    registrados = {
        str(fila["merchant_id"]): _fecha_evento(
            str(fila["event_at"])
        )
        for _, fila in relevantes.loc[
            relevantes["event_type"]
            == "REGISTERED"
        ].iterrows()
    }

    activados = {
        str(fila["merchant_id"]): _fecha_evento(
            str(fila["event_at"])
        )
        for _, fila in relevantes.loc[
            relevantes["event_type"]
            == "ACTIVATED"
        ].iterrows()
    }

    sin_registro = sorted(
        set(activados)
        - set(registrados)
    )

    if sin_registro:
        raise GeneracionVentasError(
            "Existen comercios ACTIVATED sin "
            "REGISTERED: "
            f"{', '.join(sin_registro)}."
        )

    fechas = {
        merchant_id: (
            registrados[merchant_id],
            activated_date,
        )
        for merchant_id, activated_date
        in activados.items()
    }

    if any(
        activated_date < registered_date
        for registered_date, activated_date
        in fechas.values()
    ):
        raise GeneracionVentasError(
            "ACTIVATED no puede ser anterior "
            "a REGISTERED."
        )

    return fechas


def _primera_transaccion(
    merchant_id: str,
    registered_date: date,
    activated_date: date,
    end_date: date,
    rng: np.random.Generator,
) -> date | None:
    """Selecciona conversión a tiempo, tardía o no observada."""

    if merchant_id in ANCLAS_PRIMERA_TRANSACCION:
        origen, desplazamiento = (
            ANCLAS_PRIMERA_TRANSACCION[
                merchant_id
            ]
        )

        if origen == "NONE":
            return None

        fecha_origen = (
            activated_date
            if origen == "ACTIVATED"
            else registered_date
        )

        fecha = (
            fecha_origen
            + timedelta(
                days=int(
                    desplazamiento or 0
                )
            )
        )

        return (
            fecha
            if fecha <= end_date
            else None
        )

    tipo_conversion = str(
        rng.choice(
            (
                "ON_TIME",
                "LATE",
                "NONE",
            ),
            p=(
                0.78,
                0.12,
                0.10,
            ),
        )
    )

    if tipo_conversion == "NONE":
        return None

    if tipo_conversion == "ON_TIME":
        fecha_limite = min(
            registered_date
            + timedelta(days=30),
            end_date,
        )

        if activated_date > fecha_limite:
            return None

        maximo_retraso = min(
            7,
            (
                fecha_limite
                - activated_date
            ).days,
        )

        retraso = int(
            rng.integers(
                0,
                maximo_retraso + 1,
            )
        )

        return (
            activated_date
            + timedelta(days=retraso)
        )

    fecha_inicial = max(
        activated_date,
        registered_date
        + timedelta(days=31),
    )

    fecha_limite = min(
        registered_date
        + timedelta(days=60),
        end_date,
    )

    if fecha_inicial > fecha_limite:
        return None

    retraso = int(
        rng.integers(
            0,
            (
                fecha_limite
                - fecha_inicial
            ).days
            + 1,
        )
    )

    return (
        fecha_inicial
        + timedelta(days=retraso)
    )


def _serie_base_comercio(
    merchant_id: str,
    business_segment: str,
    activated_date: date,
    first_transaction_date: date | None,
    end_date: date,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Construye un calendario denso con ceros explícitos."""

    fechas = pd.date_range(
        activated_date,
        end_date,
        freq="D",
    ).date

    if business_segment == "MEDIUM":
        probabilidad_actividad = 0.72
        promedio_transacciones = 8
        ticket_minimo = 70_000
        ticket_maximo = 260_000
    else:
        probabilidad_actividad = 0.60
        promedio_transacciones = 4
        ticket_minimo = 35_000
        ticket_maximo = 180_000

    filas: list[
        dict[str, object]
    ] = []

    for sales_date in fechas:
        approved_transactions = 0
        sales_amount_cop = 0

        if (
            first_transaction_date is not None
            and sales_date
            >= first_transaction_date
        ):
            es_primera = (
                sales_date
                == first_transaction_date
            )

            tiene_actividad = (
                es_primera
                or rng.random()
                < probabilidad_actividad
            )

            if tiene_actividad:
                approved_transactions = max(
                    1,
                    int(
                        rng.poisson(
                            promedio_transacciones
                        )
                    ),
                )

                ticket_promedio = int(
                    rng.integers(
                        ticket_minimo,
                        ticket_maximo + 1,
                    )
                )

                sales_amount_cop = (
                    approved_transactions
                    * ticket_promedio
                )

        filas.append(
            {
                "merchant_id": merchant_id,
                "sales_date": (
                    sales_date.isoformat()
                ),
                "approved_transactions": (
                    approved_transactions
                ),
                "sales_amount_cop": (
                    sales_amount_cop
                ),
            }
        )

    return pd.DataFrame(
        filas,
        columns=COLUMNAS_VENTAS_DIARIAS,
    )


def _distribuir_total_ventana(
    ventas: pd.DataFrame,
    inicio: date,
    fin: date,
    total_cop: int,
    dias_activos: int,
    ticket_objetivo: int,
    rng: np.random.Generator,
) -> None:
    """Distribuye un total sin etiquetar el escenario en raw."""

    fechas = pd.to_datetime(
        ventas["sales_date"]
    ).dt.date

    mascara = fechas.between(
        inicio,
        fin,
        inclusive="both",
    )

    indices = ventas.index[
        mascara
    ].to_numpy()

    dias_esperados = (
        fin - inicio
    ).days + 1

    if len(indices) != dias_esperados:
        raise GeneracionVentasError(
            "No existe cobertura completa para "
            "aplicar un escenario de ventana."
        )

    ventas.loc[
        indices,
        [
            "approved_transactions",
            "sales_amount_cop",
        ],
    ] = 0

    if total_cop == 0:
        return

    cantidad_activa = min(
        dias_activos,
        len(indices),
    )

    indices_activos = np.sort(
        rng.choice(
            indices,
            size=cantidad_activa,
            replace=False,
        )
    )

    pesos = rng.integers(
        1,
        101,
        size=cantidad_activa,
    )

    montos = (
        total_cop
        * pesos
        // int(pesos.sum())
    ).astype(int)

    montos[0] += (
        total_cop
        - int(montos.sum())
    )

    transacciones = np.maximum(
        1,
        np.rint(
            montos
            / ticket_objetivo
        ).astype(int),
    )

    ventas.loc[
        indices_activos,
        "approved_transactions",
    ] = transacciones

    ventas.loc[
        indices_activos,
        "sales_amount_cop",
    ] = montos


def _aplicar_tendencia_reciente(
    ventas: pd.DataFrame,
    merchant_id: str,
    business_segment: str,
    first_transaction_date: date | None,
    config: ProjectConfig,
    rng: np.random.Generator,
) -> None:
    """Inserta señales detectables en dos ventanas recientes."""

    window_days = (
        config.sales_change.window_days
    )

    observation_end = (
        config.project.as_of_date
        - timedelta(days=1)
    )

    observation_start = (
        observation_end
        - timedelta(
            days=window_days - 1
        )
    )

    baseline_end = (
        observation_start
        - timedelta(days=1)
    )

    baseline_start = (
        baseline_end
        - timedelta(
            days=window_days - 1
        )
    )

    if (
        first_transaction_date is None
        or first_transaction_date
        > baseline_start
    ):
        return

    ticket_objetivo = (
        150_000
        if business_segment == "MEDIUM"
        else 90_000
    )

    if merchant_id in ANCLAS_VENTANAS_28D:
        baseline_total, current_total = (
            ANCLAS_VENTANAS_28D[
                merchant_id
            ]
        )

        dias_base = 20

        dias_actuales = (
            0
            if current_total == 0
            else 20
        )
    else:
        fechas = pd.to_datetime(
            ventas["sales_date"]
        ).dt.date

        mascara_base = fechas.between(
            baseline_start,
            baseline_end,
            inclusive="both",
        )

        total_observado = int(
            ventas.loc[
                mascara_base,
                "sales_amount_cop",
            ].sum()
        )

        minimo_base = (
            10_000_000
            if business_segment == "MEDIUM"
            else 5_000_000
        )

        baseline_total = max(
            total_observado,
            minimo_base,
        )

        escenario = str(
            rng.choice(
                ESCENARIOS_TENDENCIA,
                p=PROBABILIDADES_TENDENCIA,
            )
        )

        if escenario == "STABLE_28D":
            factor = float(
                rng.uniform(
                    0.90,
                    1.10,
                )
            )
        elif escenario == "DECLINE_28D":
            factor = float(
                rng.uniform(
                    0.45,
                    0.75,
                )
            )
        elif escenario == "GROWTH_28D":
            factor = float(
                rng.uniform(
                    1.25,
                    1.75,
                )
            )
        else:
            factor = 0.0

        current_total = int(
            round(
                baseline_total
                * factor
                / 1_000
            )
            * 1_000
        )

        dias_base = int(
            rng.integers(
                14,
                25,
            )
        )

        dias_actuales = (
            0
            if current_total == 0
            else int(
                rng.integers(
                    14,
                    25,
                )
            )
        )

    _distribuir_total_ventana(
        ventas,
        baseline_start,
        baseline_end,
        baseline_total,
        dias_base,
        ticket_objetivo,
        rng,
    )

    _distribuir_total_ventana(
        ventas,
        observation_start,
        observation_end,
        current_total,
        dias_actuales,
        ticket_objetivo,
        rng,
    )


def generar_ventas_diarias(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Genera un calendario diario por comercio activado."""

    _validar_entradas(
        comercios,
        eventos,
        config,
    )

    fechas_onboarding = (
        _extraer_fechas_onboarding(
            eventos
        )
    )

    comercios_por_id = (
        comercios.set_index(
            "merchant_id"
        )
    )

    comercios_desconocidos = sorted(
        set(fechas_onboarding)
        - set(
            comercios_por_id
            .index
            .astype(str)
        )
    )

    if comercios_desconocidos:
        raise GeneracionVentasError(
            "Existen comercios activados fuera "
            "del catálogo: "
            f"{', '.join(comercios_desconocidos)}."
        )

    rng = _crear_generador_aleatorio(
        config.project.seed
    )

    grupos: list[
        pd.DataFrame
    ] = []

    for merchant_id in sorted(
        fechas_onboarding
    ):
        (
            registered_date,
            activated_date,
        ) = fechas_onboarding[
            merchant_id
        ]

        if (
            activated_date
            > config.project.end_date
        ):
            raise GeneracionVentasError(
                f"{merchant_id} fue activado "
                "después de end_date."
            )

        business_segment = str(
            comercios_por_id.loc[
                merchant_id,
                "business_segment",
            ]
        )

        first_transaction_date = (
            _primera_transaccion(
                merchant_id,
                registered_date,
                activated_date,
                config.project.end_date,
                rng,
            )
        )

        ventas_comercio = (
            _serie_base_comercio(
                merchant_id,
                business_segment,
                activated_date,
                first_transaction_date,
                config.project.end_date,
                rng,
            )
        )

        _aplicar_tendencia_reciente(
            ventas_comercio,
            merchant_id,
            business_segment,
            first_transaction_date,
            config,
            rng,
        )

        grupos.append(
            ventas_comercio
        )

    if not grupos:
        return pd.DataFrame(
            columns=COLUMNAS_VENTAS_DIARIAS
        )

    ventas = pd.concat(
        grupos,
        ignore_index=True,
    ).sort_values(
        [
            "merchant_id",
            "sales_date",
        ],
        kind="stable",
        ignore_index=True,
    )

    return ventas.loc[
        :,
        list(COLUMNAS_VENTAS_DIARIAS),
    ]


def guardar_ventas_diarias(
    ventas: pd.DataFrame,
    ruta: str | Path = Path(
        "data/raw/merchant_sales_daily.csv"
    ),
) -> Path:
    """Guarda las ventas diarias con bytes reproducibles."""

    ruta_salida = Path(
        ruta
    )

    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ventas.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return ruta_salida


def main() -> None:
    """Genera la fuente raw de actividad comercial."""

    config = load_config()

    comercios = pd.read_csv(
        "data/raw/merchants.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    eventos = pd.read_csv(
        "data/raw/onboarding_events.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    ventas = generar_ventas_diarias(
        comercios,
        eventos,
        config,
    )

    ruta = guardar_ventas_diarias(
        ventas
    )

    print(
        "Ventas diarias generadas: "
        f"{len(ventas)} filas para "
        f"{ventas['merchant_id'].nunique()} "
        f"comercios en {ruta}."
    )


if __name__ == "__main__":
    main()