from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from smb_merchant_lifecycle.config import (
    ProjectConfig,
    load_config,
)
from smb_merchant_lifecycle.comercios.generacion import (
    generar_comercios,
)
from smb_merchant_lifecycle.onboarding.generacion import (
    generar_eventos_onboarding,
)
from smb_merchant_lifecycle.ventas.generacion import (
    ANCLAS_VENTANAS_28D,
    COLUMNAS_VENTAS_DIARIAS,
    GeneracionVentasError,
    generar_ventas_diarias,
    guardar_ventas_diarias,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "project.toml"
)


def _generar_datos() -> tuple[
    ProjectConfig,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    config = load_config(
        CONFIG_PATH
    )

    comercios = generar_comercios(
        config
    )

    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )

    ventas = generar_ventas_diarias(
        comercios,
        eventos,
        config,
    )

    return (
        config,
        comercios,
        eventos,
        ventas,
    )


def _fechas_ventanas(
    config: ProjectConfig,
) -> tuple[
    date,
    date,
    date,
    date,
]:
    observation_end = (
        config.project.as_of_date
        - timedelta(days=1)
    )

    observation_start = (
        observation_end
        - timedelta(
            days=(
                config.sales_change.window_days
                - 1
            )
        )
    )

    baseline_end = (
        observation_start
        - timedelta(days=1)
    )

    baseline_start = (
        baseline_end
        - timedelta(
            days=(
                config.sales_change.window_days
                - 1
            )
        )
    )

    return (
        baseline_start,
        baseline_end,
        observation_start,
        observation_end,
    )


def test_esquema_grano_y_volumen() -> None:
    (
        config,
        _,
        eventos,
        ventas,
    ) = _generar_datos()

    activaciones = eventos.loc[
        eventos["event_type"]
        == "ACTIVATED"
    ].copy()

    fechas_activacion = (
        activaciones["event_at"].map(
            lambda valor: (
                datetime.fromisoformat(
                    valor
                ).date()
            )
        )
    )

    filas_esperadas = sum(
        (
            config.project.end_date
            - activated_date
        ).days
        + 1
        for activated_date
        in fechas_activacion
    )

    assert list(
        ventas.columns
    ) == list(
        COLUMNAS_VENTAS_DIARIAS
    )

    assert (
        len(ventas)
        == filas_esperadas
        == 26_737
    )

    assert (
        ventas["merchant_id"].nunique()
        == 98
    )

    assert not ventas.duplicated(
        subset=[
            "merchant_id",
            "sales_date",
        ]
    ).any()


def test_calendario_es_denso_para_cada_activado() -> None:
    (
        config,
        _,
        eventos,
        ventas,
    ) = _generar_datos()

    activaciones = {
        str(fila["merchant_id"]): (
            datetime.fromisoformat(
                str(fila["event_at"])
            ).date()
        )
        for _, fila in eventos.loc[
            eventos["event_type"]
            == "ACTIVATED"
        ].iterrows()
    }

    assert set(
        ventas["merchant_id"]
    ) == set(
        activaciones
    )

    for merchant_id, grupo in ventas.groupby(
        "merchant_id",
        sort=False,
    ):
        fechas_observadas = grupo[
            "sales_date"
        ].tolist()

        fechas_esperadas = [
            fecha.date().isoformat()
            for fecha in pd.date_range(
                activaciones[
                    str(merchant_id)
                ],
                config.project.end_date,
                freq="D",
            )
        ]

        assert (
            fechas_observadas
            == fechas_esperadas
        )


def test_conteos_y_montos_son_coherentes() -> None:
    _, _, _, ventas = _generar_datos()

    assert pd.api.types.is_integer_dtype(
        ventas["approved_transactions"]
    )

    assert pd.api.types.is_integer_dtype(
        ventas["sales_amount_cop"]
    )

    assert ventas[
        "approved_transactions"
    ].ge(0).all()

    assert ventas[
        "sales_amount_cop"
    ].ge(0).all()

    assert ventas.loc[
        ventas[
            "approved_transactions"
        ].eq(0),
        "sales_amount_cop",
    ].eq(0).all()

    assert ventas.loc[
        ventas[
            "approved_transactions"
        ].gt(0),
        "sales_amount_cop",
    ].gt(0).all()


def test_no_hay_actividad_anterior_a_activated() -> None:
    _, _, eventos, ventas = _generar_datos()

    activaciones = {
        str(fila["merchant_id"]): (
            datetime.fromisoformat(
                str(fila["event_at"])
            ).date()
        )
        for _, fila in eventos.loc[
            eventos["event_type"]
            == "ACTIVATED"
        ].iterrows()
    }

    for merchant_id, grupo in ventas.groupby(
        "merchant_id",
        sort=False,
    ):
        primera_fecha_observada = min(
            pd.to_datetime(
                grupo["sales_date"]
            ).dt.date
        )

        assert (
            primera_fecha_observada
            == activaciones[
                str(merchant_id)
            ]
        )


def test_primeras_transacciones_ancla() -> None:
    _, _, _, ventas = _generar_datos()

    primeras = (
        ventas.loc[
            ventas[
                "approved_transactions"
            ].gt(0)
        ]
        .groupby(
            "merchant_id"
        )["sales_date"]
        .min()
    )

    assert (
        primeras["M001"]
        == "2025-01-14"
    )

    assert (
        primeras["M006"]
        == "2025-04-12"
    )

    assert (
        primeras["M007"]
        == "2025-11-08"
    )

    assert (
        primeras["M008"]
        == "2026-05-18"
    )

    assert (
        primeras["M009"]
        == "2026-08-25"
    )

    assert (
        "M010"
        not in primeras.index
    )


def test_ventanas_ancla_tienen_totales_exactos() -> None:
    config, _, _, ventas = _generar_datos()

    (
        baseline_start,
        baseline_end,
        observation_start,
        observation_end,
    ) = _fechas_ventanas(
        config
    )

    ventas = ventas.assign(
        __sales_date=pd.to_datetime(
            ventas["sales_date"]
        ).dt.date
    )

    for merchant_id, (
        baseline_esperado,
        current_esperado,
    ) in ANCLAS_VENTANAS_28D.items():
        comercio = ventas.loc[
            ventas["merchant_id"]
            == merchant_id
        ]

        baseline = int(
            comercio.loc[
                comercio[
                    "__sales_date"
                ].between(
                    baseline_start,
                    baseline_end,
                    inclusive="both",
                ),
                "sales_amount_cop",
            ].sum()
        )

        current = int(
            comercio.loc[
                comercio[
                    "__sales_date"
                ].between(
                    observation_start,
                    observation_end,
                    inclusive="both",
                ),
                "sales_amount_cop",
            ].sum()
        )

        assert (
            baseline
            == baseline_esperado
        )

        assert (
            current
            == current_esperado
        )

    assert (
        100
        * (
            36_000_000
            - 50_000_000
        )
        / 50_000_000
    ) == pytest.approx(
        -28.0
    )

    assert (
        100
        * (
            42_000_000
            - 30_000_000
        )
        / 30_000_000
    ) == pytest.approx(
        40.0
    )

    assert (
        100
        * (
            0
            - 28_000_000
        )
        / 28_000_000
    ) == pytest.approx(
        -100.0
    )

    assert (
        100
        * (
            42_000_000
            - 40_000_000
        )
        / 40_000_000
    ) == pytest.approx(
        5.0
    )


def test_lineas_base_ancla_cumplen_actividad_minima() -> None:
    config, _, _, ventas = _generar_datos()

    (
        baseline_start,
        baseline_end,
        _,
        _,
    ) = _fechas_ventanas(
        config
    )

    fechas = pd.to_datetime(
        ventas["sales_date"]
    ).dt.date

    for merchant_id in ANCLAS_VENTANAS_28D:
        mascara = (
            ventas["merchant_id"].eq(
                merchant_id
            )
            & fechas.between(
                baseline_start,
                baseline_end,
                inclusive="both",
            )
        )

        baseline = ventas.loc[
            mascara
        ]

        assert int(
            baseline[
                "approved_transactions"
            ].sum()
        ) >= (
            config.sales_change
            .minimum_baseline_approved_transactions
        )

        assert int(
            baseline[
                "approved_transactions"
            ].gt(0).sum()
        ) >= (
            config.sales_change
            .minimum_baseline_active_days
        )


def test_misma_semilla_genera_mismas_ventas() -> None:
    config = load_config(
        CONFIG_PATH
    )

    comercios = generar_comercios(
        config
    )

    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )

    primera = generar_ventas_diarias(
        comercios,
        eventos,
        config,
    )

    segunda = generar_ventas_diarias(
        comercios,
        eventos,
        config,
    )

    pd.testing.assert_frame_equal(
        primera,
        segunda,
    )


def test_orden_de_entrada_no_cambia_las_ventas() -> None:
    config = load_config(
        CONFIG_PATH
    )

    comercios = generar_comercios(
        config
    )

    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )

    esperado = generar_ventas_diarias(
        comercios,
        eventos,
        config,
    )

    observado = generar_ventas_diarias(
        comercios.sample(
            frac=1,
            random_state=41,
        ).reset_index(
            drop=True
        ),
        eventos.sample(
            frac=1,
            random_state=42,
        ).reset_index(
            drop=True
        ),
        config,
    )

    pd.testing.assert_frame_equal(
        observado,
        esperado,
    )


def test_rechaza_entradas_inseguras() -> None:
    config = load_config(
        CONFIG_PATH
    )

    comercios = generar_comercios(
        config
    )

    eventos = generar_eventos_onboarding(
        comercios,
        config,
    )

    with pytest.raises(
        GeneracionVentasError,
        match="business_segment",
    ):
        generar_ventas_diarias(
            comercios.drop(
                columns=[
                    "business_segment",
                ]
            ),
            eventos,
            config,
        )

    with pytest.raises(
        GeneracionVentasError,
        match="merchant_id duplicados",
    ):
        generar_ventas_diarias(
            pd.concat(
                [
                    comercios,
                    comercios.iloc[[0]],
                ],
                ignore_index=True,
            ),
            eventos,
            config,
        )


def test_archivo_guardado_es_reproducible(
    tmp_path: Path,
) -> None:
    _, _, _, ventas = _generar_datos()

    primera_ruta = guardar_ventas_diarias(
        ventas,
        tmp_path / "primero.csv",
    )

    segunda_ruta = guardar_ventas_diarias(
        ventas,
        tmp_path / "segundo.csv",
    )

    assert (
        primera_ruta.read_bytes()
        == segunda_ruta.read_bytes()
    )

    guardado = pd.read_csv(
        primera_ruta,
        encoding="utf-8",
        dtype={
            "merchant_id": str,
            "sales_date": str,
            "approved_transactions": (
                "int64"
            ),
            "sales_amount_cop": "int64",
        },
        keep_default_na=False,
    )

    assert list(
        guardado.columns
    ) == list(
        COLUMNAS_VENTAS_DIARIAS
    )

    assert (
        len(guardado)
        == 26_737
    )