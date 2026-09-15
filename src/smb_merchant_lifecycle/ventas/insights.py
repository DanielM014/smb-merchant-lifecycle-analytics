"""Insights comerciales derivados de las métricas de ventas de 28 días."""

from __future__ import annotations

import re
from datetime import date
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.ventas.metricas_28d import (
    COLUMNAS_METRICAS_28D,
    MOTIVOS_NO_ELEGIBILIDAD,
    _fechas_ventanas,
)


TIPOS_INSIGHT_DIRECCIONAL = frozenset(
    {
        "SALES_VOLUME_DECLINE_28D",
        "SALES_VOLUME_GROWTH_28D",
    }
)

PRIORIDADES_INSIGHT = frozenset(
    {
        "HIGH",
        "MEDIUM",
        "LOW",
    }
)

COLUMNAS_INSIGHTS = (
    "insight_id",
    "merchant_id",
    "as_of_date",
    "insight_type",
    "observation_start",
    "observation_end",
    "baseline_start",
    "baseline_end",
    "current_value_cop",
    "baseline_value_cop",
    "change_pct",
    "priority",
    "evidence_text",
    "recommendation_text",
)


class InsightsComercialesError(ValueError):
    """Indica que las métricas no permiten generar insights confiables."""


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

    raise InsightsComercialesError(
        f"{campo} debe contener únicamente True o False."
    )


def _fecha_iso(valor: object, campo: str) -> date:
    """Interpreta una fecha ISO obligatoria y canónica."""

    texto = _texto(valor)

    try:
        fecha = date.fromisoformat(texto)
    except ValueError as exc:
        raise InsightsComercialesError(
            f"{campo} debe contener fechas ISO válidas."
        ) from exc

    if fecha.isoformat() != texto:
        raise InsightsComercialesError(
            f"{campo} debe usar el formato YYYY-MM-DD."
        )

    return fecha


def _entero_nullable(
    valor: object,
    campo: str,
) -> int | None:
    """Interpreta un entero no negativo o conserva un valor vacío."""

    if pd.isna(valor) or _texto(valor) == "":
        return None

    if isinstance(valor, bool):
        raise InsightsComercialesError(
            f"{campo} debe contener enteros no negativos o vacíos."
        )

    if isinstance(valor, Integral):
        numero = int(valor)
    elif isinstance(valor, Real):
        real = float(valor)
        if not isfinite(real) or not real.is_integer():
            raise InsightsComercialesError(
                f"{campo} debe contener enteros no negativos o vacíos."
            )
        numero = int(real)
    else:
        texto = _texto(valor)
        if re.fullmatch(r"0|[1-9]\d*", texto) is None:
            raise InsightsComercialesError(
                f"{campo} debe contener enteros no negativos o vacíos."
            )
        numero = int(texto)

    if numero < 0:
        raise InsightsComercialesError(
            f"{campo} debe contener enteros no negativos o vacíos."
        )

    return numero


def _decimal_nullable(
    valor: object,
    campo: str,
) -> float | None:
    """Interpreta un decimal finito o conserva un valor vacío."""

    if pd.isna(valor) or _texto(valor) == "":
        return None

    if isinstance(valor, bool):
        raise InsightsComercialesError(
            f"{campo} debe contener números finitos o valores vacíos."
        )

    try:
        numero = float(valor)
    except (TypeError, ValueError) as exc:
        raise InsightsComercialesError(
            f"{campo} debe contener números finitos o valores vacíos."
        ) from exc

    if not isfinite(numero):
        raise InsightsComercialesError(
            f"{campo} debe contener números finitos o valores vacíos."
        )

    return numero


def _prioridad(
    change_pct: float,
    config: ProjectConfig,
) -> str:
    """Clasifica una señal por la magnitud absoluta de su cambio."""

    magnitud = abs(change_pct)
    priority = config.sales_change.priority

    if magnitud >= priority.high_absolute_change_pct:
        return "HIGH"

    if magnitud >= priority.medium_absolute_change_pct:
        return "MEDIUM"

    if magnitud >= priority.low_absolute_change_pct:
        return "LOW"

    raise InsightsComercialesError(
        "Un insight direccional no puede quedar por debajo de la "
        "prioridad mínima configurada."
    )


def _preparar_metricas(
    metricas_28d: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Valida el contrato y las relaciones de la salida de 28 días."""

    if tuple(metricas_28d.columns) != COLUMNAS_METRICAS_28D:
        raise InsightsComercialesError(
            "merchant_sales_28d_metrics.csv no cumple el esquema esperado."
        )

    if metricas_28d.empty:
        raise InsightsComercialesError(
            "merchant_sales_28d_metrics.csv no puede estar vacío."
        )

    if metricas_28d.duplicated(
        subset=["merchant_id", "as_of_date"]
    ).any():
        raise InsightsComercialesError(
            "La clave merchant_id y as_of_date debe ser única."
        )

    metricas = metricas_28d.copy(deep=True)
    metricas["merchant_id"] = metricas["merchant_id"].map(_texto)

    if not metricas["merchant_id"].map(
        lambda valor: re.fullmatch(r"M\d{3}", valor) is not None
    ).all():
        raise InsightsComercialesError(
            "merchant_id debe usar el formato M000."
        )

    for campo in (
        "is_sales_change_eligible",
        "is_sales_volume_decline_28d",
        "is_sales_volume_growth_28d",
    ):
        metricas[campo] = metricas[campo].map(
            lambda valor, nombre=campo: _booleano_estricto(
                valor,
                nombre,
            )
        )

    for campo in (
        "baseline_value_cop",
        "current_value_cop",
    ):
        metricas[campo] = pd.array(
            metricas[campo].map(
                lambda valor, nombre=campo: _entero_nullable(
                    valor,
                    nombre,
                )
            ),
            dtype="Int64",
        )

    metricas["change_pct"] = pd.array(
        metricas["change_pct"].map(
            lambda valor: _decimal_nullable(
                valor,
                "change_pct",
            )
        ),
        dtype="Float64",
    )

    (
        baseline_start,
        baseline_end,
        observation_start,
        observation_end,
    ) = _fechas_ventanas(config)
    fechas_esperadas = {
        "as_of_date": config.project.as_of_date,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "observation_start": observation_start,
        "observation_end": observation_end,
    }

    for campo, esperada in fechas_esperadas.items():
        observadas = metricas[campo].map(
            lambda valor, nombre=campo: _fecha_iso(valor, nombre)
        )

        if not observadas.eq(esperada).all():
            raise InsightsComercialesError(
                f"{campo} no coincide con la configuración del corte."
            )

    for indice, fila in metricas.iterrows():
        elegible = bool(fila["is_sales_change_eligible"])
        motivo = _texto(
            fila["sales_change_ineligibility_reason"]
        )
        change_pct = fila["change_pct"]
        es_caida = bool(
            fila["is_sales_volume_decline_28d"]
        )
        es_crecimiento = bool(
            fila["is_sales_volume_growth_28d"]
        )

        if es_caida and es_crecimiento:
            raise InsightsComercialesError(
                "Una evaluación no puede ser caída y crecimiento "
                "simultáneamente."
            )

        if not elegible:
            if motivo not in MOTIVOS_NO_ELEGIBILIDAD:
                raise InsightsComercialesError(
                    "Todo comercio no elegible debe tener un motivo válido."
                )

            if pd.notna(change_pct) or es_caida or es_crecimiento:
                raise InsightsComercialesError(
                    "Un comercio no elegible no puede producir cambio "
                    "porcentual ni señales direccionales."
                )

            continue

        if motivo:
            raise InsightsComercialesError(
                "Un comercio elegible no puede tener motivo de "
                "no elegibilidad."
            )

        baseline = fila["baseline_value_cop"]
        current = fila["current_value_cop"]

        if pd.isna(baseline) or pd.isna(current) or pd.isna(change_pct):
            raise InsightsComercialesError(
                "Todo comercio elegible debe tener valores y cambio."
            )

        if int(baseline) <= 0:
            raise InsightsComercialesError(
                "La línea base de un comercio elegible debe ser positiva."
            )

        cambio_esperado = round(
            100.0 * (int(current) - int(baseline)) / int(baseline),
            6,
        )

        if float(change_pct) != cambio_esperado:
            raise InsightsComercialesError(
                "change_pct no concilia con los valores de sus ventanas "
                f"en la fila {indice + 2}."
            )

        caida_esperada = (
            cambio_esperado
            <= config.sales_change.decline_threshold_pct
        )
        crecimiento_esperado = (
            cambio_esperado
            >= config.sales_change.growth_threshold_pct
        )

        if (
            es_caida != caida_esperada
            or es_crecimiento != crecimiento_esperado
        ):
            raise InsightsComercialesError(
                "Las banderas direccionales no concilian con change_pct."
            )

    return metricas.sort_values(
        ["merchant_id", "as_of_date"],
        kind="stable",
        ignore_index=True,
    )


def _formatear_cop(valor: int) -> str:
    """Representa un entero COP con separadores de miles legibles."""

    return f"{valor:,}".replace(",", ".")


def _formatear_porcentaje(valor: float) -> str:
    """Conserva hasta seis decimales y usa coma decimal en el texto."""

    texto = f"{abs(valor):.6f}".rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def _tipo_insight(fila: pd.Series) -> str | None:
    """Traduce una única bandera direccional al tipo de insight."""

    if bool(fila["is_sales_volume_decline_28d"]):
        return "SALES_VOLUME_DECLINE_28D"

    if bool(fila["is_sales_volume_growth_28d"]):
        return "SALES_VOLUME_GROWTH_28D"

    return None


def _crear_insight_id(
    merchant_id: str,
    as_of_date: str,
    insight_type: str,
) -> str:
    """Construye un identificador semántico, estable y reproducible."""

    codigo_tipo = {
        "SALES_VOLUME_DECLINE_28D": "SVD28D",
        "SALES_VOLUME_GROWTH_28D": "SVG28D",
    }[insight_type]

    return (
        "INS-"
        f"{as_of_date.replace('-', '')}-"
        f"{merchant_id}-{codigo_tipo}"
    )


def _evidencia(
    fila: pd.Series,
    insight_type: str,
) -> str:
    """Expresa en español los valores y ventanas que sustentan la señal."""

    verbo = (
        "cayó"
        if insight_type == "SALES_VOLUME_DECLINE_28D"
        else "creció"
    )
    porcentaje = _formatear_porcentaje(
        float(fila["change_pct"])
    )
    baseline = _formatear_cop(
        int(fila["baseline_value_cop"])
    )
    current = _formatear_cop(
        int(fila["current_value_cop"])
    )

    return (
        f"El volumen aprobado {verbo} {porcentaje} %: de "
        f"{baseline} COP ({fila['baseline_start']} a "
        f"{fila['baseline_end']}) a {current} COP "
        f"({fila['observation_start']} a "
        f"{fila['observation_end']})."
    )


def _recomendacion(insight_type: str) -> str:
    """Propone seguimiento sin convertir una hipótesis en causa."""

    if insight_type == "SALES_VOLUME_DECLINE_28D":
        return (
            "Contactar al comercio para entender si existe un problema "
            "operativo, estacional o de continuidad del servicio."
        )

    return (
        "Contactar al comercio para comprender el crecimiento observado "
        "y evaluar oportunidades de acompañamiento comercial."
    )


def construir_insights_comerciales(
    metricas_28d: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Genera un insight por cada señal direccional válida."""

    metricas = _preparar_metricas(
        metricas_28d,
        config,
    )
    filas: list[dict[str, object]] = []

    for _, metrica in metricas.iterrows():
        insight_type = _tipo_insight(metrica)

        if insight_type is None:
            continue

        merchant_id = str(metrica["merchant_id"])
        as_of_date = str(metrica["as_of_date"])
        change_pct = float(metrica["change_pct"])

        filas.append(
            {
                "insight_id": _crear_insight_id(
                    merchant_id,
                    as_of_date,
                    insight_type,
                ),
                "merchant_id": merchant_id,
                "as_of_date": as_of_date,
                "insight_type": insight_type,
                "observation_start": metrica["observation_start"],
                "observation_end": metrica["observation_end"],
                "baseline_start": metrica["baseline_start"],
                "baseline_end": metrica["baseline_end"],
                "current_value_cop": int(
                    metrica["current_value_cop"]
                ),
                "baseline_value_cop": int(
                    metrica["baseline_value_cop"]
                ),
                "change_pct": change_pct,
                "priority": _prioridad(change_pct, config),
                "evidence_text": _evidencia(
                    metrica,
                    insight_type,
                ),
                "recommendation_text": _recomendacion(insight_type),
            }
        )

    insights = pd.DataFrame(
        filas,
        columns=COLUMNAS_INSIGHTS,
    )

    if not insights.empty:
        insights["current_value_cop"] = insights[
            "current_value_cop"
        ].astype("int64")
        insights["baseline_value_cop"] = insights[
            "baseline_value_cop"
        ].astype("int64")
        insights["change_pct"] = insights["change_pct"].astype(
            "float64"
        )

    if insights["insight_id"].duplicated().any():
        raise AssertionError(
            "commercial_insights.csv contiene insight_id duplicados."
        )

    if insights.duplicated(
        subset=["merchant_id", "as_of_date", "insight_type"]
    ).any():
        raise AssertionError(
            "Un comercio tiene una señal direccional duplicada al corte."
        )

    return insights


def guardar_insights_comerciales(
    insights: pd.DataFrame,
    ruta: str | Path = Path(
        "data/procesada/commercial_insights.csv"
    ),
) -> Path:
    """Guarda los insights con orden y esquema reproducibles."""

    if tuple(insights.columns) != COLUMNAS_INSIGHTS:
        raise InsightsComercialesError(
            "commercial_insights.csv no cumple el esquema esperado."
        )

    if insights["insight_id"].duplicated().any():
        raise InsightsComercialesError(
            "insight_id debe ser único en commercial_insights.csv."
        )

    tipos_invalidos = sorted(
        set(insights["insight_type"])
        - TIPOS_INSIGHT_DIRECCIONAL
    )
    prioridades_invalidas = sorted(
        set(insights["priority"])
        - PRIORIDADES_INSIGHT
    )

    if tipos_invalidos:
        raise InsightsComercialesError(
            "Existen tipos de insight inválidos: "
            f"{', '.join(tipos_invalidos)}."
        )

    if prioridades_invalidas:
        raise InsightsComercialesError(
            "Existen prioridades inválidas: "
            f"{', '.join(prioridades_invalidas)}."
        )

    ruta_salida = Path(ruta)
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    insights.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return ruta_salida


def main() -> None:
    """Genera los insights desde las métricas procesadas de 28 días."""

    config = load_config()
    metricas = pd.read_csv(
        "data/procesada/merchant_sales_28d_metrics.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    insights = construir_insights_comerciales(
        metricas,
        config,
    )
    ruta = guardar_insights_comerciales(insights)
    tipos = insights["insight_type"].value_counts()
    caidas = int(tipos.get("SALES_VOLUME_DECLINE_28D", 0))
    crecimientos = int(
        tipos.get("SALES_VOLUME_GROWTH_28D", 0)
    )

    print(
        "Insights comerciales generados: "
        f"{len(insights)} señales, {caidas} caídas y "
        f"{crecimientos} crecimientos en {ruta}."
    )


if __name__ == "__main__":
    main()
