"""Métricas de cambio entre dos ventanas comerciales de 28 días."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.ventas.contratos import (
    ContratoVentasError,
    validar_contrato_ventas,
)
from smb_merchant_lifecycle.ventas.modelo import (
    COLUMNAS_MODELO_CICLO_VIDA,
)


MOTIVOS_NO_ELEGIBILIDAD = frozenset(
    {
        "INSUFFICIENT_HISTORY",
        "ZERO_BASELINE",
        "INCOMPLETE_WINDOW",
        "INSUFFICIENT_BASELINE_ACTIVITY",
    }
)

COLUMNAS_METRICAS_28D = (
    "merchant_id",
    "as_of_date",
    "first_transaction_date",
    "days_since_first_transaction",
    "baseline_start",
    "baseline_end",
    "observation_start",
    "observation_end",
    "baseline_value_cop",
    "current_value_cop",
    "baseline_approved_transactions",
    "current_approved_transactions",
    "baseline_active_days",
    "current_active_days",
    "is_sales_change_eligible",
    "sales_change_ineligibility_reason",
    "change_pct",
    "is_sales_volume_decline_28d",
    "is_sales_volume_growth_28d",
)

_COLUMNAS_ENTERAS_NULLABLES = (
    "days_since_first_transaction",
    "baseline_value_cop",
    "current_value_cop",
    "baseline_approved_transactions",
    "current_approved_transactions",
    "baseline_active_days",
    "current_active_days",
)


class MetricasVentas28dError(ValueError):
    """Indica que las fuentes no permiten calcular métricas confiables."""


def _texto(valor: object) -> str:
    """Convierte un escalar en texto sin representar valores nulos."""

    if pd.isna(valor):
        return ""

    return str(valor).strip()


def _booleano_estricto(valor: object, campo: str) -> bool:
    """Interpreta únicamente booleanos inequívocos."""

    if isinstance(valor, bool):
        return valor

    texto = _texto(valor).lower()

    if texto == "true":
        return True
    if texto == "false":
        return False

    raise MetricasVentas28dError(
        f"{campo} debe contener únicamente True o False."
    )


def _fecha_opcional(valor: object, campo: str) -> date | None:
    """Interpreta una fecha ISO y conserva los valores vacíos."""

    texto = _texto(valor)

    if not texto:
        return None

    try:
        fecha = date.fromisoformat(texto)
    except ValueError as exc:
        raise MetricasVentas28dError(
            f"{campo} debe contener fechas ISO válidas o valores vacíos."
        ) from exc

    if fecha.isoformat() != texto:
        raise MetricasVentas28dError(
            f"{campo} debe usar el formato YYYY-MM-DD."
        )

    return fecha


def _entero_nullable(valor: object, campo: str) -> int | None:
    """Interpreta un entero opcional sin aceptar decimales truncados."""

    texto = _texto(valor)

    if not texto:
        return None

    try:
        numero = int(texto)
    except ValueError as exc:
        raise MetricasVentas28dError(
            f"{campo} debe contener enteros o valores vacíos."
        ) from exc

    if str(numero) != texto:
        raise MetricasVentas28dError(
            f"{campo} debe contener enteros o valores vacíos."
        )

    return numero


def _preparar_modelo(
    modelo_ciclo_vida: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Valida el grano y los campos usados del ciclo de vida."""

    if tuple(modelo_ciclo_vida.columns) != (
        COLUMNAS_MODELO_CICLO_VIDA
    ):
        raise MetricasVentas28dError(
            "merchant_lifecycle_bi.csv no cumple el esquema esperado."
        )

    if modelo_ciclo_vida.empty:
        raise MetricasVentas28dError(
            "merchant_lifecycle_bi.csv no puede estar vacío."
        )

    if modelo_ciclo_vida["merchant_id"].duplicated().any():
        raise MetricasVentas28dError(
            "merchant_id debe ser único en merchant_lifecycle_bi.csv."
        )

    modelo = modelo_ciclo_vida.copy(deep=True)
    modelo["merchant_id"] = modelo["merchant_id"].map(_texto)
    modelo["is_technically_activated"] = modelo[
        "is_technically_activated"
    ].map(
        lambda valor: _booleano_estricto(
            valor,
            "is_technically_activated",
        )
    )

    primeras: list[date | None] = []
    dias_observados: list[int | None] = []

    for _, fila in modelo.iterrows():
        first_date = _fecha_opcional(
            fila["first_transaction_date"],
            "first_transaction_date",
        )
        days_since = _entero_nullable(
            fila["days_since_first_transaction"],
            "days_since_first_transaction",
        )
        esperado = (
            (config.project.as_of_date - first_date).days
            if first_date is not None
            else None
        )

        if days_since != esperado:
            raise MetricasVentas28dError(
                "days_since_first_transaction no concilia con "
                "first_transaction_date y as_of_date."
            )

        primeras.append(first_date)
        dias_observados.append(days_since)

    modelo["__first_transaction_date"] = primeras
    modelo["__days_since_first_transaction"] = pd.array(
        dias_observados,
        dtype="Int64",
    )

    return modelo.sort_values(
        "merchant_id",
        kind="stable",
        ignore_index=True,
    )


def _preparar_ventas(
    ventas: pd.DataFrame,
    modelo: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Valida el contrato y las relaciones necesarias para medir."""

    try:
        ventas_validadas = validar_contrato_ventas(
            ventas.copy(deep=True),
            config,
        )
    except (
        ContratoVentasError,
        pa.errors.SchemaError,
        pa.errors.SchemaErrors,
    ) as exc:
        raise MetricasVentas28dError(
            "merchant_sales_daily.csv no cumple el contrato de ventas."
        ) from exc

    if ventas_validadas.duplicated(
        subset=["merchant_id", "sales_date"]
    ).any():
        raise MetricasVentas28dError(
            "La clave merchant_id y sales_date debe ser única."
        )

    ventas_preparadas = ventas_validadas.copy(deep=True)
    ventas_preparadas["merchant_id"] = ventas_preparadas[
        "merchant_id"
    ].map(_texto)
    ventas_preparadas["__sales_date"] = ventas_preparadas[
        "sales_date"
    ].map(date.fromisoformat)
    ventas_preparadas["approved_transactions"] = pd.to_numeric(
        ventas_preparadas["approved_transactions"],
        errors="raise",
    ).astype("int64")
    ventas_preparadas["sales_amount_cop"] = pd.to_numeric(
        ventas_preparadas["sales_amount_cop"],
        errors="raise",
    ).astype("int64")

    coherencia_ceros = ventas_preparadas[
        "approved_transactions"
    ].eq(0).eq(
        ventas_preparadas["sales_amount_cop"].eq(0)
    )

    if not coherencia_ceros.all():
        raise MetricasVentas28dError(
            "Transacciones y monto deben ser cero o positivos "
            "de manera coherente."
        )

    ids_modelo = set(modelo["merchant_id"])
    ids_ventas = set(ventas_preparadas["merchant_id"])
    ids_desconocidos = sorted(ids_ventas - ids_modelo)

    if ids_desconocidos:
        raise MetricasVentas28dError(
            "Ventas contiene comercios ausentes del ciclo de vida: "
            f"{', '.join(ids_desconocidos)}."
        )

    activacion_por_comercio = modelo.set_index(
        "merchant_id"
    )["is_technically_activated"]
    ids_no_activados = sorted(
        merchant_id
        for merchant_id in ids_ventas
        if not bool(activacion_por_comercio[merchant_id])
    )

    if ids_no_activados:
        raise MetricasVentas28dError(
            "Existen ventas para comercios no activados técnicamente: "
            f"{', '.join(ids_no_activados)}."
        )

    primeras_ventas = (
        ventas_preparadas.loc[
            ventas_preparadas["approved_transactions"].gt(0)
        ]
        .groupby("merchant_id")["__sales_date"]
        .min()
        .to_dict()
    )

    for _, fila in modelo.iterrows():
        merchant_id = str(fila["merchant_id"])
        declarada = fila["__first_transaction_date"]
        derivada = primeras_ventas.get(merchant_id)
        tiene_filas = merchant_id in ids_ventas

        if declarada is None and derivada is not None:
            raise MetricasVentas28dError(
                "first_transaction_date no concilia con ventas positivas "
                f"para {merchant_id}."
            )

        if declarada is not None and derivada is not None:
            if declarada != derivada:
                raise MetricasVentas28dError(
                    "first_transaction_date no concilia con ventas "
                    f"positivas para {merchant_id}."
                )

        if declarada is not None and derivada is None and tiene_filas:
            raise MetricasVentas28dError(
                "El ciclo de vida declara primera transacción, pero la "
                f"serie disponible de {merchant_id} no contiene actividad."
            )

    return ventas_preparadas.sort_values(
        ["merchant_id", "__sales_date"],
        kind="stable",
        ignore_index=True,
    )


def _fechas_ventanas(
    config: ProjectConfig,
) -> tuple[date, date, date, date]:
    """Deriva dos ventanas consecutivas, cerradas e inclusivas."""

    dias = config.sales_change.window_days
    observation_end = config.project.end_date
    observation_start = observation_end - timedelta(
        days=dias - 1
    )
    baseline_end = observation_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=dias - 1)

    return (
        baseline_start,
        baseline_end,
        observation_start,
        observation_end,
    )


def _resumir_ventana(
    serie: pd.DataFrame,
    inicio: date,
    fin: date,
    window_days: int,
) -> tuple[bool, int | None, int | None, int | None]:
    """Comprueba cobertura y resume una ventana solo si está completa."""

    ventana = serie.loc[
        serie["__sales_date"].between(
            inicio,
            fin,
            inclusive="both",
        )
    ]
    fechas_observadas = set(ventana["__sales_date"])
    fechas_esperadas = {
        inicio + timedelta(days=desplazamiento)
        for desplazamiento in range(window_days)
    }
    completa = fechas_observadas == fechas_esperadas

    if not completa:
        return False, None, None, None

    return (
        True,
        int(ventana["sales_amount_cop"].sum()),
        int(ventana["approved_transactions"].sum()),
        int(ventana["approved_transactions"].gt(0).sum()),
    )


def _motivo_no_elegibilidad(
    *,
    first_transaction_date: date | None,
    baseline_start: date,
    ventanas_completas: bool,
    baseline_value_cop: int | None,
    baseline_approved_transactions: int | None,
    baseline_active_days: int | None,
    config: ProjectConfig,
) -> str:
    """Aplica en orden las reglas aprobadas de elegibilidad."""

    if (
        first_transaction_date is None
        or first_transaction_date > baseline_start
    ):
        return "INSUFFICIENT_HISTORY"

    if not ventanas_completas:
        return "INCOMPLETE_WINDOW"

    if baseline_value_cop == 0:
        return "ZERO_BASELINE"

    if (
        baseline_approved_transactions
        < config.sales_change.minimum_baseline_approved_transactions
        or baseline_active_days
        < config.sales_change.minimum_baseline_active_days
    ):
        return "INSUFFICIENT_BASELINE_ACTIVITY"

    return ""


def _calcular_change_pct(
    baseline_value_cop: int,
    current_value_cop: int,
) -> float:
    """Calcula el cambio porcentual usando la ventana anterior como base."""

    return round(
        100.0
        * (current_value_cop - baseline_value_cop)
        / baseline_value_cop,
        6,
    )


def construir_metricas_28d(
    modelo_ciclo_vida: pd.DataFrame,
    ventas: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Construye una evaluación de cambio de ventas por comercio."""

    modelo = _preparar_modelo(
        modelo_ciclo_vida,
        config,
    )
    ventas_preparadas = _preparar_ventas(
        ventas,
        modelo,
        config,
    )
    series = {
        str(merchant_id): grupo.reset_index(drop=True)
        for merchant_id, grupo in ventas_preparadas.groupby(
            "merchant_id",
            sort=False,
        )
    }
    (
        baseline_start,
        baseline_end,
        observation_start,
        observation_end,
    ) = _fechas_ventanas(config)
    window_days = config.sales_change.window_days
    filas: list[dict[str, object]] = []

    for _, fila_modelo in modelo.iterrows():
        merchant_id = str(fila_modelo["merchant_id"])
        first_date = fila_modelo["__first_transaction_date"]
        serie = series.get(
            merchant_id,
            ventas_preparadas.iloc[0:0],
        )
        (
            baseline_complete,
            baseline_value,
            baseline_transactions,
            baseline_active_days,
        ) = _resumir_ventana(
            serie,
            baseline_start,
            baseline_end,
            window_days,
        )
        (
            current_complete,
            current_value,
            current_transactions,
            current_active_days,
        ) = _resumir_ventana(
            serie,
            observation_start,
            observation_end,
            window_days,
        )
        ventanas_completas = (
            baseline_complete and current_complete
        )
        motivo = _motivo_no_elegibilidad(
            first_transaction_date=first_date,
            baseline_start=baseline_start,
            ventanas_completas=ventanas_completas,
            baseline_value_cop=baseline_value,
            baseline_approved_transactions=baseline_transactions,
            baseline_active_days=baseline_active_days,
            config=config,
        )
        elegible = motivo == ""
        change_pct = (
            _calcular_change_pct(
                baseline_value,
                current_value,
            )
            if elegible
            else None
        )

        filas.append(
            {
                "merchant_id": merchant_id,
                "as_of_date": config.project.as_of_date.isoformat(),
                "first_transaction_date": (
                    first_date.isoformat()
                    if first_date is not None
                    else ""
                ),
                "days_since_first_transaction": fila_modelo[
                    "__days_since_first_transaction"
                ],
                "baseline_start": baseline_start.isoformat(),
                "baseline_end": baseline_end.isoformat(),
                "observation_start": observation_start.isoformat(),
                "observation_end": observation_end.isoformat(),
                "baseline_value_cop": baseline_value,
                "current_value_cop": current_value,
                "baseline_approved_transactions": (
                    baseline_transactions
                ),
                "current_approved_transactions": current_transactions,
                "baseline_active_days": baseline_active_days,
                "current_active_days": current_active_days,
                "is_sales_change_eligible": elegible,
                "sales_change_ineligibility_reason": motivo,
                "change_pct": change_pct,
                "is_sales_volume_decline_28d": bool(
                    elegible
                    and change_pct
                    <= config.sales_change.decline_threshold_pct
                ),
                "is_sales_volume_growth_28d": bool(
                    elegible
                    and change_pct
                    >= config.sales_change.growth_threshold_pct
                ),
            }
        )

    metricas = pd.DataFrame(
        filas,
        columns=COLUMNAS_METRICAS_28D,
    )

    for campo in _COLUMNAS_ENTERAS_NULLABLES:
        metricas[campo] = pd.array(
            metricas[campo],
            dtype="Int64",
        )

    metricas["change_pct"] = pd.array(
        metricas["change_pct"],
        dtype="Float64",
    )

    motivos_observados = set(
        metricas.loc[
            metricas["sales_change_ineligibility_reason"].ne(""),
            "sales_change_ineligibility_reason",
        ]
    )

    if not motivos_observados.issubset(MOTIVOS_NO_ELEGIBILIDAD):
        raise AssertionError(
            "El modelo produjo motivos de no elegibilidad inválidos."
        )

    if metricas.duplicated(
        subset=["merchant_id", "as_of_date"]
    ).any():
        raise AssertionError(
            "La clave merchant_id y as_of_date quedó duplicada."
        )

    return metricas


def guardar_metricas_28d(
    metricas: pd.DataFrame,
    ruta: str | Path = Path(
        "data/procesada/merchant_sales_28d_metrics.csv"
    ),
) -> Path:
    """Guarda las métricas conservando su esquema reproducible."""

    if tuple(metricas.columns) != COLUMNAS_METRICAS_28D:
        raise MetricasVentas28dError(
            "Las métricas de 28 días no cumplen el esquema esperado."
        )

    if metricas.duplicated(
        subset=["merchant_id", "as_of_date"]
    ).any():
        raise MetricasVentas28dError(
            "La clave merchant_id y as_of_date debe ser única."
        )

    ruta_salida = Path(ruta)
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    metricas.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return ruta_salida


def main() -> None:
    """Calcula las métricas desde las salidas procesadas oficiales."""

    config = load_config()
    modelo = pd.read_csv(
        "data/procesada/merchant_lifecycle_bi.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    ventas = pd.read_csv(
        "data/procesada/merchant_sales_daily.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    metricas = construir_metricas_28d(
        modelo,
        ventas,
        config,
    )
    ruta = guardar_metricas_28d(metricas)
    elegibles = int(metricas["is_sales_change_eligible"].sum())
    caidas = int(metricas["is_sales_volume_decline_28d"].sum())
    crecimientos = int(
        metricas["is_sales_volume_growth_28d"].sum()
    )

    print(
        "Métricas de ventas de 28 días generadas: "
        f"{len(metricas)} comercios, {elegibles} elegibles, "
        f"{caidas} caídas y {crecimientos} crecimientos en {ruta}."
    )


if __name__ == "__main__":
    main()
