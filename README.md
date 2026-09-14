# SMB Merchant Lifecycle & Commercial Insights

Proyecto de analítica con datos sintéticos que sigue el ciclo de vida de comercios SMB desde el registro hasta su primera transacción, analiza su comportamiento de ventas y convierte señales cuantificables en insights, recomendaciones y casos de seguimiento.

## Estado

Fase 1 — creación del repositorio y entorno reproducible.

## Alcance del primer corte

- Funnel completo de onboarding.
- Activación técnica y primera transacción.
- KPI de primera transacción dentro de 30 días.
- Actividad comercial diaria sintética.
- Comparaciones consecutivas de 28 días.
- `SALES_VOLUME_DECLINE_28D`.
- `SALES_VOLUME_GROWTH_28D`.
- Recomendaciones y casos de seguimiento.
- Controles de calidad y pruebas automatizadas.
- Tablero analítico de dos páginas.

## Principios

- Todos los datos son sintéticos.
- La misma configuración y semilla deben producir los mismos resultados.
- Las métricas deben conservar su granularidad, denominador y reglas de elegibilidad.
- Los insights describen evidencia observable y no inventan causas.
- Cada alerta direccional genera un caso de seguimiento trazable.