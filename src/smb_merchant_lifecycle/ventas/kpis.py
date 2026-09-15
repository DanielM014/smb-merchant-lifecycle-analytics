"""KPI comerciales de onboarding derivados del ciclo de vida."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.ventas.modelo import (
    COLUMNAS_MODELO_CICLO_VIDA,
    ESTADOS_PRIMERA_TRANSACCION,
)


COLUMNAS_RESUMEN_KPI = (
    "kpi_id",
    "kpi_label",
    "scope",
    "as_of_date",
    "window_days",
    "numerator_count",
    "denominator_count",
    "population_count",
    "kpi_value",
    "kpi_unit",
    "kpi_status",
)

KPI_IDS = (
    "COMMERCIAL_ACTIVATION_30D_RATE",
    "MEDIAN_DAYS_TO_FIRST_TRANSACTION_30D",
    "ACTIVE_30D_RATE",
)


class ResumenKpiError(ValueError):
    """Indica que el modelo no permite calcular KPI confiables."""


def _texto(valor: object) -> str:
    """Convierte un escalar en texto sin representar nulos."""

    if pd.isna(valor):
        return ""

    return str(valor).strip()


def _booleano_estricto(valor: object, campo: str) -> bool:
    """Interpreta únicamente True o False."""

    if isinstance(valor, bool):
        return valor

    texto = _texto(valor).lower()

    if texto == "true":
        return True
    if texto == "false":
        return False

    raise ResumenKpiError(
        f"{campo} debe contener únicamente True o False."
    )


def _booleano_nullable(
    valor: object,
    campo: str,
) -> bool | None:
    """Interpreta booleanos y conserva vacíos como no evaluados."""

    texto = _texto(valor).lower()

    if not texto:
        return None
    if texto == "true":
        return True
    if texto == "false":
        return False

    raise ResumenKpiError(
        f"{campo} debe contener True, False o un valor vacío."
    )


def _entero_nullable(
    serie: pd.Series,
    campo: str,
) -> pd.Series:
    """Normaliza una columna entera que puede estar vacía."""

    normalizada = serie.map(_texto).replace("", pd.NA)

    try:
        return pd.to_numeric(
            normalizada,
            errors="raise",
        ).astype("Int64")
    except (TypeError, ValueError) as exc:
        raise ResumenKpiError(
            f"{campo} debe contener enteros o valores vacíos."
        ) from exc


def _preparar_modelo(
    modelo_ciclo_vida: pd.DataFrame,
) -> pd.DataFrame:
    """Valida el esquema y las relaciones entre campos KPI."""

    if tuple(modelo_ciclo_vida.columns) != (
        COLUMNAS_MODELO_CICLO_VIDA
    ):
        raise ResumenKpiError(
            "merchant_lifecycle_bi.csv no cumple el esquema esperado."
        )

    if modelo_ciclo_vida.empty:
        raise ResumenKpiError(
            "merchant_lifecycle_bi.csv no puede estar vacío."
        )

    if modelo_ciclo_vida["merchant_id"].duplicated().any():
        raise ResumenKpiError(
            "merchant_id debe ser único en merchant_lifecycle_bi.csv."
        )

    modelo = modelo_ciclo_vida.copy(deep=True)

    for campo in (
        "is_technically_activated",
        "is_commercial_activation_30d_eligible",
        "is_active_30d_eligible",
    ):
        modelo[campo] = modelo[campo].map(
            lambda valor, nombre=campo: _booleano_estricto(
                valor,
                nombre,
            )
        )

    for campo in (
        "commercial_activation_30d",
        "is_active_30d",
    ):
        modelo[campo] = pd.array(
            modelo[campo].map(
                lambda valor, nombre=campo: _booleano_nullable(
                    valor,
                    nombre,
                )
            ),
            dtype="boolean",
        )

    modelo["days_to_first_transaction"] = _entero_nullable(
        modelo["days_to_first_transaction"],
        "days_to_first_transaction",
    )

    estados_invalidos = sorted(
        set(modelo["first_transaction_status"])
        - ESTADOS_PRIMERA_TRANSACCION
    )

    if estados_invalidos:
        raise ResumenKpiError(
            "Existen first_transaction_status inválidos: "
            f"{', '.join(estados_invalidos)}."
        )

    elegibles_comerciales = modelo[
        "is_commercial_activation_30d_eligible"
    ]
    resultado_comercial = modelo[
        "commercial_activation_30d"
    ]

    if resultado_comercial.loc[elegibles_comerciales].isna().any():
        raise ResumenKpiError(
            "Todo comercio elegible debe tener un resultado comercial."
        )

    if resultado_comercial.loc[~elegibles_comerciales].notna().any():
        raise ResumenKpiError(
            "Un comercio no elegible no puede clasificarse como éxito "
            "o fracaso comercial."
        )

    activacion_esperada = modelo[
        "first_transaction_status"
    ].eq("ON_TIME")

    if not resultado_comercial.loc[
        elegibles_comerciales
    ].astype(bool).equals(
        activacion_esperada.loc[elegibles_comerciales]
    ):
        raise ResumenKpiError(
            "commercial_activation_30d no coincide con el estado "
            "de primera transacción."
        )

    elegibles_actividad = modelo["is_active_30d_eligible"]
    resultado_actividad = modelo["is_active_30d"]

    if resultado_actividad.loc[elegibles_actividad].isna().any():
        raise ResumenKpiError(
            "Todo comercio elegible para ACTIVE_30D debe tener resultado."
        )

    if resultado_actividad.loc[~elegibles_actividad].notna().any():
        raise ResumenKpiError(
            "Un comercio sin ventana completa no puede clasificarse "
            "en ACTIVE_30D."
        )

    con_primera = modelo["first_transaction_date"].map(
        _texto
    ).ne("")
    con_dias = modelo["days_to_first_transaction"].notna()

    if not con_primera.equals(con_dias):
        raise ResumenKpiError(
            "first_transaction_date y days_to_first_transaction "
            "no concilian."
        )

    return modelo


def _tasa(
    numerador: int,
    denominador: int,
) -> tuple[float | None, str]:
    """Calcula una tasa protegida frente a denominadores vacíos."""

    if denominador == 0:
        return None, "NOT_AVAILABLE"

    return (
        round(
            100.0 * numerador / denominador,
            6,
        ),
        "AVAILABLE",
    )


def construir_resumen_kpi(
    modelo_ciclo_vida: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Calcula los tres KPI comerciales globales de onboarding."""

    modelo = _preparar_modelo(modelo_ciclo_vida)
    elegibles_comerciales = modelo[
        "is_commercial_activation_30d_eligible"
    ]
    activados_30d = modelo[
        "commercial_activation_30d"
    ].fillna(False)
    numerador_activacion = int(
        (elegibles_comerciales & activados_30d).sum()
    )
    denominador_activacion = int(
        elegibles_comerciales.sum()
    )
    tasa_activacion, estado_activacion = _tasa(
        numerador_activacion,
        denominador_activacion,
    )

    poblacion_mediana = modelo.loc[
        elegibles_comerciales & activados_30d,
        "days_to_first_transaction",
    ]
    mediana = (
        float(poblacion_mediana.median())
        if not poblacion_mediana.empty
        else None
    )
    estado_mediana = (
        "AVAILABLE"
        if mediana is not None
        else "NOT_AVAILABLE"
    )

    elegibles_actividad = modelo["is_active_30d_eligible"]
    activos_30d = modelo["is_active_30d"].fillna(False)
    numerador_actividad = int(
        (elegibles_actividad & activos_30d).sum()
    )
    denominador_actividad = int(
        elegibles_actividad.sum()
    )
    tasa_actividad, estado_actividad = _tasa(
        numerador_actividad,
        denominador_actividad,
    )

    filas = [
        {
            "kpi_id": "COMMERCIAL_ACTIVATION_30D_RATE",
            "kpi_label": (
                "Tasa de primera transacción a 30 días"
            ),
            "scope": "GLOBAL",
            "as_of_date": config.project.as_of_date.isoformat(),
            "window_days": (
                config.onboarding.first_transaction_window_days
            ),
            "numerator_count": numerador_activacion,
            "denominator_count": denominador_activacion,
            "population_count": denominador_activacion,
            "kpi_value": tasa_activacion,
            "kpi_unit": "PCT",
            "kpi_status": estado_activacion,
        },
        {
            "kpi_id": (
                "MEDIAN_DAYS_TO_FIRST_TRANSACTION_30D"
            ),
            "kpi_label": (
                "Mediana de días hasta primera transacción a tiempo"
            ),
            "scope": "GLOBAL",
            "as_of_date": config.project.as_of_date.isoformat(),
            "window_days": (
                config.onboarding.first_transaction_window_days
            ),
            "numerator_count": None,
            "denominator_count": None,
            "population_count": len(poblacion_mediana),
            "kpi_value": mediana,
            "kpi_unit": "DAYS",
            "kpi_status": estado_mediana,
        },
        {
            "kpi_id": "ACTIVE_30D_RATE",
            "kpi_label": (
                "Tasa de actividad posterior a primera transacción"
            ),
            "scope": "GLOBAL",
            "as_of_date": config.project.as_of_date.isoformat(),
            "window_days": (
                config.onboarding.first_transaction_window_days
            ),
            "numerator_count": numerador_actividad,
            "denominator_count": denominador_actividad,
            "population_count": denominador_actividad,
            "kpi_value": tasa_actividad,
            "kpi_unit": "PCT",
            "kpi_status": estado_actividad,
        },
    ]
    resumen = pd.DataFrame(
        filas,
        columns=COLUMNAS_RESUMEN_KPI,
    )

    for campo in (
        "numerator_count",
        "denominator_count",
        "population_count",
    ):
        resumen[campo] = pd.array(
            resumen[campo],
            dtype="Int64",
        )

    resumen["kpi_value"] = pd.array(
        resumen["kpi_value"],
        dtype="Float64",
    )

    return resumen


def guardar_resumen_kpi(
    resumen: pd.DataFrame,
    ruta: str | Path = Path(
        "data/procesada/kpi_summary.csv"
    ),
) -> Path:
    """Guarda los KPI en un archivo reproducible."""

    if tuple(resumen.columns) != COLUMNAS_RESUMEN_KPI:
        raise ResumenKpiError(
            "El resumen de KPI no cumple el esquema esperado."
        )

    ruta_salida = Path(ruta)
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    resumen.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return ruta_salida


def main() -> None:
    """Calcula los KPI desde el modelo integrado procesado."""

    config = load_config()
    modelo = pd.read_csv(
        "data/procesada/merchant_lifecycle_bi.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    resumen = construir_resumen_kpi(
        modelo,
        config,
    )
    ruta = guardar_resumen_kpi(resumen)

    print(
        f"Resumen de KPI generado: "
        f"{len(resumen)} indicadores en {ruta}."
    )


if __name__ == "__main__":
    main()
