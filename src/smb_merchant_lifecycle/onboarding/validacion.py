"""Puerta de calidad para comercios y eventos de onboarding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Callable

import pandas as pd
import pandera.pandas as pa

from smb_merchant_lifecycle.config import ProjectConfig, load_config
from smb_merchant_lifecycle.onboarding.contratos import (
    COLUMNAS_COMERCIOS,
    COLUMNAS_EVENTOS_ONBOARDING,
    validar_contrato_comercios,
    validar_contrato_eventos_onboarding,
)
from smb_merchant_lifecycle.onboarding.generacion import TRANSICIONES_PERMITIDAS


COLUMNAS_VIOLACIONES = (
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

COLUMNAS_RECHAZADOS = (
    "source_table",
    "source_row_number",
    "record_id",
    "merchant_id",
    "violation_count",
    "rule_ids",
    "record_json",
)

EVENTOS_TERMINALES = frozenset({"ACTIVATED", "REJECTED", "ABANDONED"})
_COLUMNA_FILA = "__source_row_number"
_ORDEN_TABLA = {"merchants.csv": 0, "onboarding_events.csv": 1}

_REGLAS_PANDERA_EXACTAS = {
    (
        "merchants.csv",
        "merchant_id",
        "merchant_id no puede estar vacío",
    ): "MERCHANT_REQUIRED_VALUE",
    (
        "merchants.csv",
        "merchant_id",
        "merchant_id debe usar el formato M000",
    ): "MERCHANT_ID_FORMAT",
    (
        "merchants.csv",
        "merchant_name",
        "merchant_name no puede estar vacío",
    ): "MERCHANT_REQUIRED_VALUE",
    (
        "onboarding_events.csv",
        "onboarding_event_id",
        "onboarding_event_id no puede estar vacío",
    ): "ONBOARDING_REQUIRED_VALUE",
    (
        "onboarding_events.csv",
        "onboarding_event_id",
        "onboarding_event_id debe usar el formato OE0000",
    ): "ONBOARDING_EVENT_ID_FORMAT",
    (
        "onboarding_events.csv",
        "merchant_id",
        "merchant_id no puede estar vacío",
    ): "ONBOARDING_REQUIRED_VALUE",
    (
        "onboarding_events.csv",
        "merchant_id",
        "merchant_id debe usar el formato M000",
    ): "ONBOARDING_MERCHANT_ID_FORMAT",
    (
        "onboarding_events.csv",
        "event_at",
        "event_at no puede estar vacío",
    ): "ONBOARDING_REQUIRED_VALUE",
    (
        "onboarding_events.csv",
        "event_at",
        "event_at debe ser ISO 8601, usar UTC-05:00 y estar dentro del periodo",
    ): "ONBOARDING_EVENT_AT_INVALID",
}

_REGLAS_DOMINIO = {
    ("merchants.csv", "city"): "MERCHANT_CITY_INVALID",
    ("merchants.csv", "business_segment"): "MERCHANT_SEGMENT_INVALID",
    ("merchants.csv", "acquisition_channel"): "MERCHANT_CHANNEL_INVALID",
    ("onboarding_events.csv", "event_type"): "ONBOARDING_EVENT_TYPE_INVALID",
}

_MENSAJES_REGLA = {
    "MERCHANT_REQUIRED_VALUE": "El campo obligatorio debe contener texto no vacío.",
    "MERCHANT_ID_FORMAT": "merchant_id debe usar el formato M000.",
    "MERCHANT_ID_DUPLICATE": "merchant_id aparece en más de una fila.",
    "MERCHANT_CITY_INVALID": "city está fuera del dominio permitido.",
    "MERCHANT_SEGMENT_INVALID": (
        "business_segment está fuera del dominio permitido."
    ),
    "MERCHANT_CHANNEL_INVALID": (
        "acquisition_channel está fuera del dominio permitido."
    ),
    "ONBOARDING_REQUIRED_VALUE": (
        "El campo obligatorio debe contener texto no vacío."
    ),
    "ONBOARDING_EVENT_ID_FORMAT": (
        "onboarding_event_id debe usar el formato OE0000."
    ),
    "ONBOARDING_EVENT_ID_DUPLICATE": (
        "onboarding_event_id aparece en más de una fila."
    ),
    "ONBOARDING_MERCHANT_ID_FORMAT": (
        "merchant_id debe usar el formato M000."
    ),
    "ONBOARDING_EVENT_TYPE_INVALID": (
        "event_type está fuera del dominio permitido."
    ),
    "ONBOARDING_EVENT_AT_INVALID": (
        "event_at debe ser ISO 8601, usar UTC-05:00 y estar dentro del periodo."
    ),
    "COLUMN_TYPE_INVALID": "El tipo de dato no cumple el contrato.",
    "PANDERA_ROW_INVALID": "La fila incumple una regla del contrato.",
}


@dataclass(frozen=True, slots=True)
class ResultadoValidacion:
    """Contiene las salidas válidas y la evidencia de los rechazos."""

    comercios_validos: pd.DataFrame
    eventos_validos: pd.DataFrame
    violaciones: pd.DataFrame
    registros_rechazados: pd.DataFrame


def _preparar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    """Crea una copia con números de fila equivalentes al CSV."""

    preparados = datos.reset_index(drop=True).copy()
    preparados[_COLUMNA_FILA] = range(2, len(preparados) + 2)
    return preparados


def _texto(valor: object) -> str:
    """Convierte un valor escalar en texto seguro para reportes."""

    if pd.isna(valor):
        return ""
    return str(valor)


def _record_id(fila: pd.Series, campo_id: str) -> str:
    """Obtiene la clave o crea una referencia estable a la fila."""

    valor = _texto(fila[campo_id]).strip()
    if valor:
        return valor
    return f"ROW_{int(fila[_COLUMNA_FILA]):05d}"


def _agregar_violacion(
    violaciones: list[dict[str, object]],
    *,
    source_table: str,
    source_row_number: int,
    record_id: str,
    merchant_id: str,
    rule_id: str,
    field_name: str,
    message: str,
    observed_value: object,
) -> None:
    """Agrega una violación con el contrato común del reporte."""

    violaciones.append(
        {
            "source_table": source_table,
            "source_row_number": int(source_row_number),
            "record_id": record_id,
            "merchant_id": merchant_id,
            "rule_id": rule_id,
            "severity": "ERROR",
            "field_name": field_name,
            "message": message,
            "observed_value": _texto(observed_value),
        }
    )


def _normalizar_regla_pandera(
    source_table: str,
    field_name: str,
    check: str,
) -> str:
    """Convierte identificadores de Pandera en reglas estables del proyecto."""

    exacta = _REGLAS_PANDERA_EXACTAS.get((source_table, field_name, check))
    if exacta is not None:
        return exacta

    prefijo = "MERCHANT" if source_table == "merchants.csv" else "ONBOARDING"
    if check == "field_uniqueness":
        return f"{prefijo}_ID_DUPLICATE"
    if check == "not_nullable":
        return f"{prefijo}_REQUIRED_VALUE"
    if check.startswith("isin("):
        return _REGLAS_DOMINIO[(source_table, field_name)]
    if check.startswith("dtype("):
        return "COLUMN_TYPE_INVALID"
    return "PANDERA_ROW_INVALID"


def _capturar_errores_pandera(
    datos: pd.DataFrame,
    columnas: tuple[str, ...],
    validar: Callable[[pd.DataFrame], pd.DataFrame],
    source_table: str,
    campo_id: str,
    violaciones: list[dict[str, object]],
) -> None:
    """Ejecuta el contrato y traduce sus fallos de fila al reporte."""

    try:
        validar(datos.loc[:, list(columnas)])
        return
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        fallos = exc.failure_cases

    if not isinstance(fallos, pd.DataFrame) or "index" not in fallos:
        raise RuntimeError(
            f"{source_table} produjo un error de contrato no atribuible a filas."
        )

    for fallo in fallos.to_dict(orient="records"):
        indice = fallo.get("index")
        if pd.isna(indice):
            raise RuntimeError(
                f"{source_table} produjo un error de contrato sin índice de fila."
            )

        fila = datos.loc[int(indice)]
        field_name = _texto(fallo.get("column"))
        check = _texto(fallo.get("check"))
        rule_id = _normalizar_regla_pandera(
            source_table,
            field_name,
            check,
        )
        _agregar_violacion(
            violaciones,
            source_table=source_table,
            source_row_number=int(fila[_COLUMNA_FILA]),
            record_id=_record_id(fila, campo_id),
            merchant_id=_texto(fila["merchant_id"]).strip(),
            rule_id=rule_id,
            field_name=field_name,
            message=_MENSAJES_REGLA[rule_id],
            observed_value=fallo.get("failure_case"),
        )


def _filas_invalidas(
    violaciones: list[dict[str, object]],
    source_table: str,
) -> set[int]:
    """Devuelve los números de fila afectados en una tabla."""

    return {
        int(violacion["source_row_number"])
        for violacion in violaciones
        if violacion["source_table"] == source_table
    }


def _validar_integridad_referencial(
    eventos: pd.DataFrame,
    merchant_ids_validos: set[str],
    violaciones: list[dict[str, object]],
) -> None:
    """Detecta eventos cuyo comercio padre no existe o es inválido."""

    filas_con_error = _filas_invalidas(violaciones, "onboarding_events.csv")
    for _, fila in eventos.iterrows():
        numero_fila = int(fila[_COLUMNA_FILA])
        merchant_id = _texto(fila["merchant_id"]).strip()
        if numero_fila in filas_con_error or not merchant_id:
            continue
        if merchant_id in merchant_ids_validos:
            continue

        _agregar_violacion(
            violaciones,
            source_table="onboarding_events.csv",
            source_row_number=numero_fila,
            record_id=_record_id(fila, "onboarding_event_id"),
            merchant_id=merchant_id,
            rule_id="ONBOARDING_MERCHANT_NOT_FOUND_OR_INVALID",
            field_name="merchant_id",
            message="El evento no tiene un comercio padre válido.",
            observed_value=merchant_id,
        )


def _validar_secuencias(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    violaciones: list[dict[str, object]],
) -> None:
    """Valida rutas completas usando únicamente filas candidatas."""

    filas_comercio_invalidas = _filas_invalidas(violaciones, "merchants.csv")
    filas_evento_invalidas = _filas_invalidas(
        violaciones,
        "onboarding_events.csv",
    )
    comercios_candidatos = comercios.loc[
        ~comercios[_COLUMNA_FILA].isin(filas_comercio_invalidas)
    ]
    eventos_candidatos = eventos.loc[
        ~eventos[_COLUMNA_FILA].isin(filas_evento_invalidas)
    ].copy()
    eventos_candidatos["__event_datetime"] = eventos_candidatos[
        "event_at"
    ].map(datetime.fromisoformat)

    eventos_por_comercio = {
        str(merchant_id): grupo.copy()
        for merchant_id, grupo in eventos_candidatos.groupby(
            "merchant_id",
            sort=False,
        )
    }

    for _, comercio in comercios_candidatos.iterrows():
        merchant_id = str(comercio["merchant_id"])
        grupo = eventos_por_comercio.get(
            merchant_id,
            eventos_candidatos.iloc[0:0],
        )
        cantidad_registered = int(
            (grupo["event_type"] == "REGISTERED").sum()
        )
        if cantidad_registered == 1:
            continue

        _agregar_violacion(
            violaciones,
            source_table="merchants.csv",
            source_row_number=int(comercio[_COLUMNA_FILA]),
            record_id=merchant_id,
            merchant_id=merchant_id,
            rule_id="MERCHANT_REGISTERED_COUNT_INVALID",
            field_name="merchant_id",
            message=(
                "El comercio debe tener exactamente un evento REGISTERED válido."
            ),
            observed_value=cantidad_registered,
        )

    for merchant_id, grupo in eventos_por_comercio.items():
        ordenado = grupo.sort_values(
            ["__event_datetime", "onboarding_event_id", _COLUMNA_FILA]
        )
        registered = ordenado.loc[ordenado["event_type"] == "REGISTERED"]

        if len(registered) == 1:
            fecha_registered = registered.iloc[0]["__event_datetime"]
            anteriores = ordenado.loc[
                ordenado["__event_datetime"] < fecha_registered
            ]
            for _, fila in anteriores.iterrows():
                _agregar_violacion(
                    violaciones,
                    source_table="onboarding_events.csv",
                    source_row_number=int(fila[_COLUMNA_FILA]),
                    record_id=_record_id(fila, "onboarding_event_id"),
                    merchant_id=merchant_id,
                    rule_id="ONBOARDING_EVENT_BEFORE_REGISTERED",
                    field_name="event_at",
                    message="El evento ocurre antes del único REGISTERED.",
                    observed_value=fila["event_at"],
                )

        tipos_repetidos = set(
            ordenado.loc[
                ordenado["event_type"].duplicated(keep=False),
                "event_type",
            ]
        )
        for event_type in sorted(tipos_repetidos):
            for _, fila in ordenado.loc[
                ordenado["event_type"] == event_type
            ].iterrows():
                _agregar_violacion(
                    violaciones,
                    source_table="onboarding_events.csv",
                    source_row_number=int(fila[_COLUMNA_FILA]),
                    record_id=_record_id(fila, "onboarding_event_id"),
                    merchant_id=merchant_id,
                    rule_id="ONBOARDING_STAGE_REPEATED",
                    field_name="event_type",
                    message=f"La etapa {event_type} aparece más de una vez.",
                    observed_value=event_type,
                )

        filas_ordenadas = [fila for _, fila in ordenado.iterrows()]
        for anterior, actual in pairwise(filas_ordenadas):
            if anterior["__event_datetime"] >= actual["__event_datetime"]:
                _agregar_violacion(
                    violaciones,
                    source_table="onboarding_events.csv",
                    source_row_number=int(actual[_COLUMNA_FILA]),
                    record_id=_record_id(actual, "onboarding_event_id"),
                    merchant_id=merchant_id,
                    rule_id="ONBOARDING_TIMESTAMP_NOT_INCREASING",
                    field_name="event_at",
                    message="Las fechas deben ser estrictamente crecientes.",
                    observed_value=actual["event_at"],
                )

            tipo_anterior = str(anterior["event_type"])
            tipo_actual = str(actual["event_type"])
            if tipo_actual not in TRANSICIONES_PERMITIDAS[tipo_anterior]:
                _agregar_violacion(
                    violaciones,
                    source_table="onboarding_events.csv",
                    source_row_number=int(actual[_COLUMNA_FILA]),
                    record_id=_record_id(actual, "onboarding_event_id"),
                    merchant_id=merchant_id,
                    rule_id="ONBOARDING_TRANSITION_INVALID",
                    field_name="event_type",
                    message=(
                        f"La transición {tipo_anterior} -> {tipo_actual} "
                        "no está permitida."
                    ),
                    observed_value=tipo_actual,
                )

            if tipo_anterior in EVENTOS_TERMINALES:
                _agregar_violacion(
                    violaciones,
                    source_table="onboarding_events.csv",
                    source_row_number=int(actual[_COLUMNA_FILA]),
                    record_id=_record_id(actual, "onboarding_event_id"),
                    merchant_id=merchant_id,
                    rule_id="ONBOARDING_EVENT_AFTER_TERMINAL",
                    field_name="event_type",
                    message=(
                        f"No puede existir un evento después de {tipo_anterior}."
                    ),
                    observed_value=tipo_actual,
                )


def _poner_rutas_en_cuarentena(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    violaciones: list[dict[str, object]],
) -> None:
    """Rechaza una ruta completa cuando alguna de sus filas es inválida."""

    merchant_ids_invalidos = {
        str(violacion["merchant_id"])
        for violacion in violaciones
        if str(violacion["merchant_id"])
    }
    filas_comercio_invalidas = _filas_invalidas(violaciones, "merchants.csv")
    for _, fila in comercios.iterrows():
        numero_fila = int(fila[_COLUMNA_FILA])
        merchant_id = _texto(fila["merchant_id"]).strip()
        if (
            merchant_id not in merchant_ids_invalidos
            or numero_fila in filas_comercio_invalidas
        ):
            continue

        _agregar_violacion(
            violaciones,
            source_table="merchants.csv",
            source_row_number=numero_fila,
            record_id=_record_id(fila, "merchant_id"),
            merchant_id=merchant_id,
            rule_id="MERCHANT_ONBOARDING_ROUTE_INVALID",
            field_name="merchant_id",
            message="El comercio tiene una ruta de onboarding inválida.",
            observed_value=merchant_id,
        )

    filas_evento_invalidas = _filas_invalidas(
        violaciones,
        "onboarding_events.csv",
    )
    for _, fila in eventos.iterrows():
        numero_fila = int(fila[_COLUMNA_FILA])
        merchant_id = _texto(fila["merchant_id"]).strip()
        if (
            merchant_id not in merchant_ids_invalidos
            or numero_fila in filas_evento_invalidas
        ):
            continue

        _agregar_violacion(
            violaciones,
            source_table="onboarding_events.csv",
            source_row_number=numero_fila,
            record_id=_record_id(fila, "onboarding_event_id"),
            merchant_id=merchant_id,
            rule_id="ONBOARDING_ROUTE_QUARANTINED",
            field_name="merchant_id",
            message="La ruta completa se rechaza para evitar un historial parcial.",
            observed_value=merchant_id,
        )


def _crear_dataframe_violaciones(
    violaciones: list[dict[str, object]],
) -> pd.DataFrame:
    """Deduplica, ordena y asigna identificadores a las violaciones."""

    if not violaciones:
        return pd.DataFrame(columns=COLUMNAS_VIOLACIONES)

    resultado = pd.DataFrame(violaciones).drop_duplicates(
        subset=["source_table", "source_row_number", "rule_id"]
    )
    resultado["__table_order"] = resultado["source_table"].map(_ORDEN_TABLA)
    resultado = resultado.sort_values(
        ["__table_order", "source_row_number", "rule_id"],
        kind="stable",
    ).reset_index(drop=True)
    resultado.insert(
        0,
        "violation_id",
        [f"QV{numero:05d}" for numero in range(1, len(resultado) + 1)],
    )
    return resultado.loc[:, list(COLUMNAS_VIOLACIONES)]


def _valor_json(valor: object) -> object:
    """Normaliza escalares antes de serializar el registro original."""

    if pd.isna(valor):
        return None
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def _crear_registros_rechazados(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    violaciones: pd.DataFrame,
) -> pd.DataFrame:
    """Crea una fila por registro rechazado con todas sus reglas."""

    if violaciones.empty:
        return pd.DataFrame(columns=COLUMNAS_RECHAZADOS)

    grupos = {
        (str(tabla), int(numero_fila)): grupo
        for (tabla, numero_fila), grupo in violaciones.groupby(
            ["source_table", "source_row_number"],
            sort=False,
        )
    }
    rechazados: list[dict[str, object]] = []
    fuentes = (
        ("merchants.csv", comercios, COLUMNAS_COMERCIOS, "merchant_id"),
        (
            "onboarding_events.csv",
            eventos,
            COLUMNAS_EVENTOS_ONBOARDING,
            "onboarding_event_id",
        ),
    )

    for source_table, datos, columnas, campo_id in fuentes:
        for _, fila in datos.iterrows():
            numero_fila = int(fila[_COLUMNA_FILA])
            clave = (source_table, numero_fila)
            if clave not in grupos:
                continue

            grupo = grupos[clave]
            registro = {
                columna: _valor_json(fila[columna])
                for columna in columnas
            }
            rechazados.append(
                {
                    "source_table": source_table,
                    "source_row_number": numero_fila,
                    "record_id": _record_id(fila, campo_id),
                    "merchant_id": _texto(fila["merchant_id"]).strip(),
                    "violation_count": int(len(grupo)),
                    "rule_ids": "|".join(sorted(set(grupo["rule_id"]))),
                    "record_json": json.dumps(
                        registro,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

    return pd.DataFrame(rechazados, columns=COLUMNAS_RECHAZADOS)


def _separar_validos(
    datos: pd.DataFrame,
    columnas: tuple[str, ...],
    rechazados: pd.DataFrame,
    source_table: str,
) -> pd.DataFrame:
    """Retira de una tabla las filas presentes en el registro de rechazos."""

    filas_rechazadas = set(
        rechazados.loc[
            rechazados["source_table"] == source_table,
            "source_row_number",
        ]
    )
    return datos.loc[
        ~datos[_COLUMNA_FILA].isin(filas_rechazadas),
        list(columnas),
    ].reset_index(drop=True)


def _comprobar_conciliacion(
    raw: pd.DataFrame,
    validos: pd.DataFrame,
    rechazados: pd.DataFrame,
    source_table: str,
) -> None:
    """Exige que cada fila raw quede clasificada una sola vez."""

    cantidad_rechazada = int(
        (rechazados["source_table"] == source_table).sum()
    )
    if len(validos) + cantidad_rechazada != len(raw):
        raise AssertionError(f"La conciliación de {source_table} no cierra.")


def validar_integridad_onboarding(
    comercios: pd.DataFrame,
    eventos: pd.DataFrame,
    config: ProjectConfig,
) -> ResultadoValidacion:
    """Valida, reconcilia y separa las entradas de onboarding."""

    comercios_preparados = _preparar_datos(comercios)
    eventos_preparados = _preparar_datos(eventos)
    violaciones: list[dict[str, object]] = []

    _capturar_errores_pandera(
        comercios_preparados,
        COLUMNAS_COMERCIOS,
        validar_contrato_comercios,
        "merchants.csv",
        "merchant_id",
        violaciones,
    )
    filas_comercio_invalidas = _filas_invalidas(violaciones, "merchants.csv")
    merchant_ids_validos = set(
        comercios_preparados.loc[
            ~comercios_preparados[_COLUMNA_FILA].isin(
                filas_comercio_invalidas
            ),
            "merchant_id",
        ].astype(str)
    )

    _capturar_errores_pandera(
        eventos_preparados,
        COLUMNAS_EVENTOS_ONBOARDING,
        lambda datos: validar_contrato_eventos_onboarding(datos, config),
        "onboarding_events.csv",
        "onboarding_event_id",
        violaciones,
    )
    _validar_integridad_referencial(
        eventos_preparados,
        merchant_ids_validos,
        violaciones,
    )
    _validar_secuencias(
        comercios_preparados,
        eventos_preparados,
        violaciones,
    )
    _poner_rutas_en_cuarentena(
        comercios_preparados,
        eventos_preparados,
        violaciones,
    )

    dataframe_violaciones = _crear_dataframe_violaciones(violaciones)
    rechazados = _crear_registros_rechazados(
        comercios_preparados,
        eventos_preparados,
        dataframe_violaciones,
    )
    comercios_validos = _separar_validos(
        comercios_preparados,
        COLUMNAS_COMERCIOS,
        rechazados,
        "merchants.csv",
    )
    eventos_validos = _separar_validos(
        eventos_preparados,
        COLUMNAS_EVENTOS_ONBOARDING,
        rechazados,
        "onboarding_events.csv",
    )

    _comprobar_conciliacion(
        comercios_preparados,
        comercios_validos,
        rechazados,
        "merchants.csv",
    )
    _comprobar_conciliacion(
        eventos_preparados,
        eventos_validos,
        rechazados,
        "onboarding_events.csv",
    )
    validar_contrato_comercios(comercios_validos)
    validar_contrato_eventos_onboarding(eventos_validos, config)

    if not set(eventos_validos["merchant_id"]).issubset(
        set(comercios_validos["merchant_id"])
    ):
        raise AssertionError("La salida válida contiene eventos huérfanos.")

    return ResultadoValidacion(
        comercios_validos=comercios_validos,
        eventos_validos=eventos_validos,
        violaciones=dataframe_violaciones,
        registros_rechazados=rechazados,
    )


def _guardar_csv(datos: pd.DataFrame, ruta: Path) -> Path:
    """Guarda un CSV con formato reproducible."""

    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(
        ruta,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    return ruta


def guardar_resultado_validacion(
    resultado: ResultadoValidacion,
    directorio_procesada: str | Path = Path("data/procesada"),
    directorio_rechazada: str | Path = Path("data/rechazada"),
) -> dict[str, Path]:
    """Guarda datos válidos, violaciones y registros rechazados."""

    procesada = Path(directorio_procesada)
    rechazada = Path(directorio_rechazada)
    return {
        "merchants": _guardar_csv(
            resultado.comercios_validos,
            procesada / "merchants.csv",
        ),
        "onboarding_events": _guardar_csv(
            resultado.eventos_validos,
            procesada / "onboarding_events.csv",
        ),
        "quality_violations": _guardar_csv(
            resultado.violaciones,
            rechazada / "quality_violations.csv",
        ),
        "rejected_records": _guardar_csv(
            resultado.registros_rechazados,
            rechazada / "rejected_records.csv",
        ),
    }


def main() -> None:
    """Valida los archivos raw y guarda sus salidas separadas."""

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
    resultado = validar_integridad_onboarding(comercios, eventos, config)
    guardar_resultado_validacion(resultado)

    print(
        "Validación completada: "
        f"{len(resultado.comercios_validos)} comercios válidos, "
        f"{len(resultado.eventos_validos)} eventos válidos, "
        f"{len(resultado.registros_rechazados)} registros rechazados y "
        f"{len(resultado.violaciones)} violaciones."
    )


if __name__ == "__main__":
    main()
