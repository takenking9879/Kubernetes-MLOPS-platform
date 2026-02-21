# Métricas Prometheus eliminadas — kafka_main.py

## Contexto del cambio

Al consolidar la arquitectura a **Spark como router puro** (sin modo `online=False`),
la métrica de latencia de preprocesamiento en Spark quedó sin uso y fue eliminada.
Este documento sirve como referencia para reemplazarla por métricas más relevantes
a la arquitectura actual.

---

## Métrica eliminada

### `LATENCY_PREPROCESS` (Gauge)

| Campo | Valor |
|-------|-------|
| **Definida en** | `src/prometheus/` (importada en `kafka_main.py`) |
| **Tipo** | `Gauge` |
| **Medía** | Tiempo de ejecución del DSL pipeline en Spark (transformers + estimators) |
| **Última línea de uso** | `process_batch()` → `LATENCY_PREPROCESS.set(preprocess_latency)` |
| **Por qué se eliminó** | En la nueva arquitectura Spark no hace preprocessing. El preprocesamiento ocurre en Ray Serve vía `NumpyPipelineExecutor`. La métrica siempre era `0.0` en el modo que se usaba (`online=True`). |

---

## Métricas sugeridas para reemplazar

El router Spark ya no mide preprocessing, pero hay nuevas latencias relevantes
que actualmente no se instrumentan:

| Métrica sugerida | Tipo | Qué mediría | Dónde instrumentar |
|-----------------|------|-------------|-------------------|
| `LATENCY_SCHEMA_CONVERSION` | Summary/Gauge | Tiempo de `convert_schema()` — `from_json` + cast de columnas | `process_batch()`, antes/después de `convert_schema()` |
| `LATENCY_KAFKA_WRITE` | Summary/Gauge | Tiempo de `_write_to_kafka()` — escritura al topic de salida | `process_batch()`, antes/después de `_write_to_kafka()` |
| `LATENCY_JOIN` | Gauge | Tiempo del `batch_df.join(predictions_df)` | `process_batch()`, antes/después del join |
| `BATCH_EMPTY_TOTAL` | Counter | Número de micro-batches vacíos | `process_batch()`, en el bloque `if record_count == 0` |
| `KAFKA_LAG_RECORDS` | Gauge | Registros pendientes en el topic de entrada (consumer lag) | Requiere leer metadata de Kafka (Consumer API) |

### Nota sobre `LATENCY_INFERENCE`

`LATENCY_INFERENCE` ya existe y mide el round-trip HTTP completo a Ray Serve
(incluyendo DSL NumPy + predicción del modelo). Si se desea mayor granularidad,
considerar separar en:

- `LATENCY_RAY_HTTP` — solo el overhead de red (TCP + HTTP overhead)
- `LATENCY_RAY_PREPROCESSING` — tiempo del DSL NumPy en Ray Serve (requiere
  instrumentación en `src/serve/runtime.py`)
- `LATENCY_RAY_INFERENCE` — solo la inferencia del modelo PyTorch/XGBoost
  (requiere instrumentación en `src/serve/runtime.py`)

---

## Métricas actuales que siguen activas

Para referencia, estas métricas NO fueron eliminadas y siguen operativas:

| Métrica | Tipo | Descripción |
|---------|------|-------------|
| `LATENCY_INFERENCE` | Gauge | Round-trip HTTP a Ray Serve (schema convert → response) |
| `LATENCY_TOTAL_BATCH` | Gauge | Latencia total del micro-batch (start → Kafka write done) |
| `BATCH_RECORDS_TOTAL` | Gauge | Número de registros en el batch actual |
| `BATCH_ERRORS_TOTAL` | Counter | Total de micro-batches fallidos |
| `INFERENCE_LATENCY_SUMMARY` | Summary | Distribución histórica de latencias de inferencia |
| `PREDICTIONS_BY_CLASS_TOTAL` | Counter (labeled) | Predicciones acumuladas por clase (0..N) |
| `PREDICTIONS_BY_CLASS_LAST_BATCH` | Gauge (labeled) | Predicciones por clase en el último batch |
| `LATENCY_KAFKA_WRITE` | Gauge | Tiempo de `_write_to_kafka()` — escritura al topic de salida |

> `LATENCY_KAFKA_WRITE` fue implementada como reemplazo directo de `LATENCY_PREPROCESS` en el panel
> "Batch Time Decomposition" del dashboard Grafana (`mlops-inference.yaml`). Mide el tiempo
> que Spark tarda en escribir los resultados al topic de salida Kafka.
