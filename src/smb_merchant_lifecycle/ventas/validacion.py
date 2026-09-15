"""Puerta de calidad para la actividad comercial diaria."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.ventas.contratos import (
    COLUMNAS_VENTAS_DIARIAS,
    validar_contrato_ventas,
    validar_estructura_ventas,
)


TABLA_VENTAS = "merchant_sales_daily.csv"
_COLUMNA_FILA = "__source_row_number"

COLUMNAS_VIOLACIONES_VENTAS = (
    "violation_id",
    "source_table",
    "source_row_number",
    "record_id",
    "merchant_id",
    "rule_id",
    "severity",
    "field_name",
    "message",
    "observed_value",
)

COLUMNAS_RECHAZADOS_VENTAS = (
    "source_table",
    "source_row_number",
    "record_id",
    "merchant_id",
    "violation_count",
    "rule_ids",
    "record_json",
)

_MENSAJES_REGLA = {
    "SALES_REQUIRED_VALUE": (
        "El campo obligatorio debe contener un valor válido."
    ),
    "SALES_MERCHANT_ID_FORMAT": (
        "merchant_id debe usar el formato M000."
    ),
    "SALES_DATE_INVALID": (
        "sales_date debe ser YYYY-MM-DD y estar dentro del periodo."
    ),
    "SALES_APPROVED_TRANSACTIONS_INVALID": (
        "approved_transactions debe ser un entero no negativo."
    ),
    "SALES_AMOUNT_INVALID": (
        "sales_amount_cop debe ser un entero no negativo."
    ),
    "SALES_GRAIN_DUPLICATE": (
        "La clave compuesta merchant_id y sales_date está duplicada."
    ),
    "SALES_ZERO_AMOUNT_INCONSISTENT": (
        "Un día sin transacciones debe tener monto cero y un día "
        "con transacciones debe tener monto positivo."
    ),
    "SALES_MERCHANT_NOT_FOUND": (
        "El comercio de la venta no existe en el modelo de onboarding."
    ),
    "SALES_MERCHANT_NOT_ACTIVATED": (
        "Solo un comercio técnicamente activado puede tener ventas."
    ),
    "SALES_BEFORE_ACTIVATION": (
        "sales_date no puede ser anterior a la activación técnica."
    ),
    "SALES_DATE_MISSING": (
        "Falta una fecha esperada en el calendario diario del comercio."
    ),
    "SALES_SERIES_QUARANTINED": (
        "La fila se excluye porque la serie del comercio tiene una "
        "brecha de cobertura."
    ),
    "SALES_CONTRACT_ROW_INVALID": (
        "La fila incumple una regla del contrato de ventas."
    ),
}


class ValidacionVentasError(ValueError):
    """Indica que una fuente de referencia no es segura."""


@dataclass(frozen=True, slots=True)
class ResultadoValidacionVentas:
    """Contiene ventas válidas y evidencia de cuarentena."""

    ventas_validas: pd.DataFrame
    violaciones: pd.DataFrame
    registros_rechazados: pd.DataFrame


def _preparar_datos(ventas: pd.DataFrame) -> pd.DataFrame:
    """Crea una copia con números de fila equivalentes al CSV."""

    preparados = ventas.reset_index(drop=True).copy()
    preparados[_COLUMNA_FILA] = range(
        2,
        len(preparados) + 2,
    )
    return preparados


def _texto(valor: object) -> str:
    """Convierte un valor escalar en texto seguro."""

    if pd.isna(valor):
        return ""
    return str(valor)


def _record_id(fila: pd.Series) -> str:
    """Construye la clave lógica o una referencia estable a la fila."""

    merchant_id = _texto(
        fila["merchant_id"]
    ).strip()
    sales_date = _texto(
        fila["sales_date"]
    ).strip()

    if merchant_id and sales_date:
        return f"{merchant_id}|{sales_date}"

    return (
        "ROW_"
        f"{int(fila[_COLUMNA_FILA]):05d}"
    )


def _agregar_violacion(
    violaciones: list[dict[str, object]],
    *,
    source_row_number: int | None,
    record_id: str,
    merchant_id: str,
    rule_id: str,
    field_name: str,
    observed_value: object,
) -> None:
    """Agrega una violación con identificadores estables."""

    violaciones.append(
        {
            "source_table": TABLA_VENTAS,
            "source_row_number": source_row_number,
            "record_id": record_id,
            "merchant_id": merchant_id,
            "rule_id": rule_id,
            "severity": "ERROR",
            "field_name": field_name,
            "message": _MENSAJES_REGLA[rule_id],
            "observed_value": _texto(observed_value),
        }
    )


def _normalizar_regla_pandera(
    field_name: str,
    check: str,
) -> str:
    """Convierte fallos de Pandera en reglas del proyecto."""

    if check == "not_nullable" or "no puede estar vacío" in check:
        return "SALES_REQUIRED_VALUE"

    if field_name == "merchant_id" and "formato M000" in check:
        return "SALES_MERCHANT_ID_FORMAT"

    if field_name == "sales_date":
        return "SALES_DATE_INVALID"

    if field_name == "approved_transactions":
        return "SALES_APPROVED_TRANSACTIONS_INVALID"

    if field_name == "sales_amount_cop":
        return "SALES_AMOUNT_INVALID"

    return "SALES_CONTRACT_ROW_INVALID"


def _capturar_errores_contrato(
    ventas: pd.DataFrame,
    config: ProjectConfig,
    violaciones: list[dict[str, object]],
) -> None:
    """Ejecuta el contrato y traduce sus fallos al reporte."""

    try:
        validar_contrato_ventas(
            ventas.loc[
                :,
                list(COLUMNAS_VENTAS_DIARIAS),
            ],
            config,
        )
        return
    except (
        pa.errors.SchemaError,
        pa.errors.SchemaErrors,
    ) as exc:
        fallos = exc.failure_cases

    if (
        not isinstance(fallos, pd.DataFrame)
        or "index" not in fallos
    ):
        raise RuntimeError(
            "El contrato de ventas produjo un error no "
            "atribuible a filas."
        )

    for fallo in fallos.to_dict(orient="records"):
        indice = fallo.get("index")

        if pd.isna(indice):
            raise RuntimeError(
                "El contrato de ventas produjo un error "
                "sin índice de fila."
            )

        fila = ventas.loc[int(indice)]
        field_name = _texto(
            fallo.get("column")
        )
        check = _texto(
            fallo.get("check")
        )
        rule_id = _normalizar_regla_pandera(
            field_name,
            check,
        )

        _agregar_violacion(
            violaciones,
            source_row_number=int(
                fila[_COLUMNA_FILA]
            ),
            record_id=_record_id(fila),
            merchant_id=_texto(
                fila["merchant_id"]
            ).strip(),
            rule_id=rule_id,
            field_name=field_name,
            observed_value=fallo.get(
                "failure_case"
            ),
        )


def _filas_invalidas(
    violaciones: list[dict[str, object]],
) -> set[int]:
    """Devuelve los números de filas físicas con errores."""

    return {
        int(numero)
        for violacion in violaciones
        if (
            (numero := violacion["source_row_number"])
            is not None
            and not pd.isna(numero)
        )
    }


def _validar_grano(
    ventas: pd.DataFrame,
    violaciones: list[dict[str, object]],
) -> None:
    """Exige una fila por comercio y fecha."""

    invalidas = _filas_invalidas(violaciones)
    candidatas = ventas.loc[
        ~ventas[_COLUMNA_FILA].isin(invalidas)
    ]
    duplicadas = candidatas.duplicated(
        subset=["merchant_id", "sales_date"],
        keep=False,
    )

    for _, fila in candidatas.loc[
        duplicadas
    ].iterrows():
        _agregar_violacion(
            violaciones,
            source_row_number=int(
                fila[_COLUMNA_FILA]
            ),
            record_id=_record_id(fila),
            merchant_id=_texto(
                fila["merchant_id"]
            ).strip(),
            rule_id="SALES_GRAIN_DUPLICATE",
            field_name="merchant_id|sales_date",
            observed_value=_record_id(fila),
        )


def _validar_coherencia_montos(
    ventas: pd.DataFrame,
    violaciones: list[dict[str, object]],
) -> None:
    """Comprueba la relación entre transacciones y monto."""

    invalidas = _filas_invalidas(violaciones)

    for _, fila in ventas.loc[
        ~ventas[_COLUMNA_FILA].isin(invalidas)
    ].iterrows():
        transacciones = int(
            fila["approved_transactions"]
        )
        monto = int(
            fila["sales_amount_cop"]
        )

        if (transacciones == 0) == (monto == 0):
            continue

        _agregar_violacion(
            violaciones,
            source_row_number=int(
                fila[_COLUMNA_FILA]
            ),
            record_id=_record_id(fila),
            merchant_id=_texto(
                fila["merchant_id"]
            ).strip(),
            rule_id="SALES_ZERO_AMOUNT_INCONSISTENT",
            field_name=(
                "approved_transactions|sales_amount_cop"
            ),
            observed_value=(
                f"{transacciones}|{monto}"
            ),
        )


def _booleano_estricto(valor: object) -> bool:
    """Interpreta únicamente booleanos inequívocos."""

    if isinstance(valor, bool):
        return valor

    texto = _texto(valor).strip().lower()

    if texto == "true":
        return True
    if texto == "false":
        return False

    raise ValidacionVentasError(
        "is_technically_activated debe contener "
        "únicamente True o False."
    )


def _fecha_activacion(
    valor: object,
    config: ProjectConfig,
) -> date:
    """Convierte un timestamp de activación previamente modelado."""

    texto = _texto(valor).strip()

    try:
        timestamp = datetime.fromisoformat(
            texto
        )
    except ValueError as exc:
        raise ValidacionVentasError(
            "activated_at debe ser un timestamp ISO válido."
        ) from exc

    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset()
        != timedelta(hours=-5)
    ):
        raise ValidacionVentasError(
            "activated_at debe usar UTC-05:00."
        )

    fecha = timestamp.date()

    if not (
        config.project.start_date
        <= fecha
        <= config.project.end_date
    ):
        raise ValidacionVentasError(
            "activated_at está fuera del periodo configurado."
        )

    return fecha


def _preparar_modelo_activacion(
    modelo: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[set[str], dict[str, date]]:
    """Valida la referencia y extrae activaciones técnicas."""

    columnas_requeridas = {
        "merchant_id",
        "activated_at",
        "is_technically_activated",
    }
    faltantes = sorted(
        columnas_requeridas
        - set(modelo.columns)
    )

    if faltantes:
        raise ValidacionVentasError(
            "Faltan columnas del modelo de onboarding: "
            f"{', '.join(faltantes)}."
        )

    if modelo["merchant_id"].duplicated().any():
        raise ValidacionVentasError(
            "merchant_id debe ser único en el modelo "
            "de onboarding."
        )

    merchant_ids: set[str] = set()
    activaciones: dict[str, date] = {}

    for _, fila in modelo.iterrows():
        merchant_id = _texto(
            fila["merchant_id"]
        ).strip()

        if not merchant_id:
            raise ValidacionVentasError(
                "El modelo de onboarding contiene un "
                "merchant_id vacío."
            )

        merchant_ids.add(merchant_id)
        activado = _booleano_estricto(
            fila["is_technically_activated"]
        )
        activated_at = _texto(
            fila["activated_at"]
        ).strip()

        if activado:
            if not activated_at:
                raise ValidacionVentasError(
                    f"{merchant_id} está activado pero no "
                    "tiene activated_at."
                )

            activaciones[merchant_id] = (
                _fecha_activacion(
                    activated_at,
                    config,
                )
            )
        elif activated_at:
            raise ValidacionVentasError(
                f"{merchant_id} no está activado pero "
                "tiene activated_at."
            )

    return merchant_ids, activaciones


def _validar_integridad_operativa(
    ventas: pd.DataFrame,
    merchant_ids: set[str],
    activaciones: dict[str, date],
    violaciones: list[dict[str, object]],
) -> None:
    """Relaciona cada venta con un comercio activado."""

    invalidas = _filas_invalidas(violaciones)

    for _, fila in ventas.loc[
        ~ventas[_COLUMNA_FILA].isin(invalidas)
    ].iterrows():
        merchant_id = _texto(
            fila["merchant_id"]
        ).strip()
        numero_fila = int(
            fila[_COLUMNA_FILA]
        )
        record_id = _record_id(fila)

        if merchant_id not in merchant_ids:
            _agregar_violacion(
                violaciones,
                source_row_number=numero_fila,
                record_id=record_id,
                merchant_id=merchant_id,
                rule_id="SALES_MERCHANT_NOT_FOUND",
                field_name="merchant_id",
                observed_value=merchant_id,
            )
            continue

        if merchant_id not in activaciones:
            _agregar_violacion(
                violaciones,
                source_row_number=numero_fila,
                record_id=record_id,
                merchant_id=merchant_id,
                rule_id="SALES_MERCHANT_NOT_ACTIVATED",
                field_name="merchant_id",
                observed_value=merchant_id,
            )
            continue

        fecha_venta = date.fromisoformat(
            _texto(fila["sales_date"])
        )

        if fecha_venta < activaciones[merchant_id]:
            _agregar_violacion(
                violaciones,
                source_row_number=numero_fila,
                record_id=record_id,
                merchant_id=merchant_id,
                rule_id="SALES_BEFORE_ACTIVATION",
                field_name="sales_date",
                observed_value=fecha_venta.isoformat(),
            )


def _rango_fechas(
    inicio: date,
    fin: date,
) -> set[date]:
    """Construye un conjunto cerrado de fechas diarias."""

    cantidad = (fin - inicio).days + 1
    return {
        inicio + timedelta(days=desplazamiento)
        for desplazamiento in range(cantidad)
    }


def _validar_cobertura(
    ventas: pd.DataFrame,
    activaciones: dict[str, date],
    config: ProjectConfig,
    violaciones: list[dict[str, object]],
) -> set[str]:
    """Detecta fechas ausentes en cada calendario esperado."""

    invalidas = _filas_invalidas(violaciones)
    candidatas = ventas.loc[
        ~ventas[_COLUMNA_FILA].isin(invalidas)
    ]
    observadas: dict[str, set[date]] = {}

    for _, fila in candidatas.iterrows():
        merchant_id = _texto(
            fila["merchant_id"]
        ).strip()

        if merchant_id not in activaciones:
            continue

        observadas.setdefault(
            merchant_id,
            set(),
        ).add(
            date.fromisoformat(
                _texto(fila["sales_date"])
            )
        )

    comercios_en_cuarentena: set[str] = set()

    for merchant_id, fecha_activacion in sorted(
        activaciones.items()
    ):
        esperadas = _rango_fechas(
            fecha_activacion,
            config.project.end_date,
        )
        faltantes = sorted(
            esperadas
            - observadas.get(merchant_id, set())
        )

        if faltantes:
            comercios_en_cuarentena.add(
                merchant_id
            )

        for fecha_faltante in faltantes:
            texto_fecha = fecha_faltante.isoformat()
            _agregar_violacion(
                violaciones,
                source_row_number=None,
                record_id=(
                    f"{merchant_id}|{texto_fecha}"
                ),
                merchant_id=merchant_id,
                rule_id="SALES_DATE_MISSING",
                field_name="sales_date",
                observed_value="<MISSING>",
            )

    return comercios_en_cuarentena


def _poner_series_en_cuarentena(
    ventas: pd.DataFrame,
    comercios_en_cuarentena: set[str],
    violaciones: list[dict[str, object]],
) -> None:
    """Excluye la serie completa cuando su cobertura no es densa."""

    if not comercios_en_cuarentena:
        return

    invalidas = _filas_invalidas(violaciones)

    for _, fila in ventas.loc[
        ventas["merchant_id"].isin(
            comercios_en_cuarentena
        )
        & ~ventas[_COLUMNA_FILA].isin(
            invalidas
        )
    ].iterrows():
        _agregar_violacion(
            violaciones,
            source_row_number=int(
                fila[_COLUMNA_FILA]
            ),
            record_id=_record_id(fila),
            merchant_id=_texto(
                fila["merchant_id"]
            ).strip(),
            rule_id="SALES_SERIES_QUARANTINED",
            field_name="merchant_id",
            observed_value=_texto(
                fila["merchant_id"]
            ).strip(),
        )


def _crear_dataframe_violaciones(
    violaciones: list[dict[str, object]],
) -> pd.DataFrame:
    """Deduplica, ordena y asigna IDs a las violaciones."""

    if not violaciones:
        return pd.DataFrame(
            columns=COLUMNAS_VIOLACIONES_VENTAS
        )

    resultado = pd.DataFrame(
        violaciones
    ).drop_duplicates(
        subset=[
            "source_row_number",
            "record_id",
            "rule_id",
            "field_name",
        ]
    )
    resultado["__row_order"] = (
        resultado["source_row_number"]
        .fillna(10**12)
        .astype("int64")
    )
    resultado = resultado.sort_values(
        [
            "merchant_id",
            "__row_order",
            "record_id",
            "rule_id",
        ],
        kind="stable",
    ).reset_index(drop=True)
    resultado.insert(
        0,
        "violation_id",
        [
            f"SQV{numero:05d}"
            for numero in range(
                1,
                len(resultado) + 1,
            )
        ],
    )
    resultado["source_row_number"] = pd.array(
        resultado["source_row_number"],
        dtype="Int64",
    )

    return resultado.loc[
        :,
        list(COLUMNAS_VIOLACIONES_VENTAS),
    ]


def _valor_json(valor: object) -> object:
    """Normaliza escalares antes de serializar la fila raw."""

    if pd.isna(valor):
        return None
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def _crear_registros_rechazados(
    ventas: pd.DataFrame,
    violaciones: pd.DataFrame,
) -> pd.DataFrame:
    """Crea una fila por registro físico rechazado."""

    if violaciones.empty:
        return pd.DataFrame(
            columns=COLUMNAS_RECHAZADOS_VENTAS
        )

    violaciones_fisicas = violaciones.dropna(
        subset=["source_row_number"]
    )
    grupos = {
        int(numero_fila): grupo
        for numero_fila, grupo
        in violaciones_fisicas.groupby(
            "source_row_number",
            sort=False,
        )
    }
    rechazados: list[dict[str, object]] = []

    for _, fila in ventas.iterrows():
        numero_fila = int(
            fila[_COLUMNA_FILA]
        )

        if numero_fila not in grupos:
            continue

        grupo = grupos[numero_fila]
        registro = {
            columna: _valor_json(
                fila[columna]
            )
            for columna in COLUMNAS_VENTAS_DIARIAS
        }
        rechazados.append(
            {
                "source_table": TABLA_VENTAS,
                "source_row_number": numero_fila,
                "record_id": _record_id(fila),
                "merchant_id": _texto(
                    fila["merchant_id"]
                ).strip(),
                "violation_count": int(
                    len(grupo)
                ),
                "rule_ids": "|".join(
                    sorted(
                        set(
                            grupo[
                                "rule_id"
                            ].astype(str)
                        )
                    )
                ),
                "record_json": json.dumps(
                    registro,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )

    return pd.DataFrame(
        rechazados,
        columns=COLUMNAS_RECHAZADOS_VENTAS,
    )


def _separar_ventas_validas(
    ventas: pd.DataFrame,
    rechazados: pd.DataFrame,
) -> pd.DataFrame:
    """Retira filas en cuarentena y normaliza medidas numéricas."""

    filas_rechazadas = set(
        rechazados["source_row_number"]
    )
    validas = ventas.loc[
        ~ventas[_COLUMNA_FILA].isin(
            filas_rechazadas
        ),
        list(COLUMNAS_VENTAS_DIARIAS),
    ].copy()
    validas["approved_transactions"] = (
        pd.to_numeric(
            validas["approved_transactions"],
            errors="raise",
        ).astype("int64")
    )
    validas["sales_amount_cop"] = (
        pd.to_numeric(
            validas["sales_amount_cop"],
            errors="raise",
        ).astype("int64")
    )

    return validas.sort_values(
        ["merchant_id", "sales_date"],
        kind="stable",
        ignore_index=True,
    )


def _comprobar_salida(
    ventas_raw: pd.DataFrame,
    ventas_validas: pd.DataFrame,
    rechazados: pd.DataFrame,
    config: ProjectConfig,
) -> None:
    """Exige conciliación y contrato en la salida procesada."""

    if (
        len(ventas_validas)
        + len(rechazados)
        != len(ventas_raw)
    ):
        raise AssertionError(
            "La conciliación de ventas no cierra."
        )

    if ventas_validas.duplicated(
        subset=["merchant_id", "sales_date"]
    ).any():
        raise AssertionError(
            "La salida válida conserva claves duplicadas."
        )

    validar_contrato_ventas(
        ventas_validas,
        config,
    )


def validar_integridad_ventas(
    ventas: pd.DataFrame,
    modelo_onboarding: pd.DataFrame,
    config: ProjectConfig,
) -> ResultadoValidacionVentas:
    """Valida, reconcilia y separa la actividad comercial."""

    validar_estructura_ventas(ventas)
    ventas_preparadas = _preparar_datos(
        ventas
    )
    merchant_ids, activaciones = (
        _preparar_modelo_activacion(
            modelo_onboarding,
            config,
        )
    )
    violaciones: list[dict[str, object]] = []

    _capturar_errores_contrato(
        ventas_preparadas,
        config,
        violaciones,
    )
    _validar_grano(
        ventas_preparadas,
        violaciones,
    )
    _validar_coherencia_montos(
        ventas_preparadas,
        violaciones,
    )
    _validar_integridad_operativa(
        ventas_preparadas,
        merchant_ids,
        activaciones,
        violaciones,
    )
    comercios_en_cuarentena = (
        _validar_cobertura(
            ventas_preparadas,
            activaciones,
            config,
            violaciones,
        )
    )
    _poner_series_en_cuarentena(
        ventas_preparadas,
        comercios_en_cuarentena,
        violaciones,
    )

    dataframe_violaciones = (
        _crear_dataframe_violaciones(
            violaciones
        )
    )
    rechazados = _crear_registros_rechazados(
        ventas_preparadas,
        dataframe_violaciones,
    )
    ventas_validas = _separar_ventas_validas(
        ventas_preparadas,
        rechazados,
    )

    _comprobar_salida(
        ventas_preparadas,
        ventas_validas,
        rechazados,
        config,
    )

    return ResultadoValidacionVentas(
        ventas_validas=ventas_validas,
        violaciones=dataframe_violaciones,
        registros_rechazados=rechazados,
    )


def _guardar_csv(
    datos: pd.DataFrame,
    ruta: Path,
) -> Path:
    """Guarda un CSV con formato reproducible."""

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    datos.to_csv(
        ruta,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    return ruta


def guardar_resultado_validacion_ventas(
    resultado: ResultadoValidacionVentas,
    directorio_procesada: str | Path = Path(
        "data/procesada"
    ),
    directorio_rechazada: str | Path = Path(
        "data/rechazada"
    ),
) -> dict[str, Path]:
    """Guarda ventas válidas y evidencia de cuarentena."""

    procesada = Path(directorio_procesada)
    rechazada = Path(directorio_rechazada)

    return {
        "merchant_sales_daily": _guardar_csv(
            resultado.ventas_validas,
            procesada / "merchant_sales_daily.csv",
        ),
        "sales_quality_violations": _guardar_csv(
            resultado.violaciones,
            rechazada / "sales_quality_violations.csv",
        ),
        "sales_rejected_records": _guardar_csv(
            resultado.registros_rechazados,
            rechazada / "sales_rejected_records.csv",
        ),
    }


def main() -> None:
    """Valida el raw de ventas contra el onboarding procesado."""

    config = load_config()
    ventas = pd.read_csv(
        "data/raw/merchant_sales_daily.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    modelo_onboarding = pd.read_csv(
        "data/procesada/merchant_onboarding_bi.csv",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    resultado = validar_integridad_ventas(
        ventas,
        modelo_onboarding,
        config,
    )
    guardar_resultado_validacion_ventas(
        resultado
    )
    brechas = int(
        resultado.violaciones["rule_id"]
        .eq("SALES_DATE_MISSING")
        .sum()
    ) if not resultado.violaciones.empty else 0

    print(
        "Validación de ventas completada: "
        f"{len(resultado.ventas_validas)} filas válidas, "
        f"{len(resultado.registros_rechazados)} "
        "registros rechazados, "
        f"{len(resultado.violaciones)} violaciones y "
        f"{brechas} brechas de cobertura."
    )


if __name__ == "__main__":
    main()
