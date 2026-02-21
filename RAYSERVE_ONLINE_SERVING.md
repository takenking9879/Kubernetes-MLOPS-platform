# Real-Time Inference con Ray Serve + NumPy DSL Executor

## Resumen de cambios

| Archivo | Acción | Por qué |
|---|---|---|
| `src/dsl/numpy_executor.py` | **Nuevo** | Ejecuta el DSL YAML row-by-row sin Spark |
| `src/converters/raw_to_features.py` | **Nuevo** | Réplica Python de `kafka_to_schema_features` (usada en tests) |
| `src/serve/pipeline_loader.py` | **Nuevo** | Descarga artifacts desde S3 vía MLflow → pyiceberg → boto3 |
| `src/serve/runtime.py` | **Modificado** | Soporte para payload `raw` + `set_executor()` |
| `src/serve/config.py` | **Modificado** | Añade campo `online: bool` a `ServingConfig` |
| `k3s/kuberay/serving/app.py` | **Modificado** | Modo `online: true/false` controlado por params.yaml |
| `k3s/spark/inference/kafka_main.py` | **Modificado** | Modo `online: true` — solo conversión de schema en Spark |
| `k3s/params.yaml` | **Modificado** | Añade `kuberay.serving.online` |
| `notebooks/test_numpy_pipeline.ipynb` | **Nuevo** | Valida NumPy vs Spark con `2026_01_01_001.parquet` |

### Sin cambios

- `src/dsl/` — transformers, estimators, pipeline, base: **intactos**
- `dsl_001.yaml` — sigue siendo el único punto de verdad

---

## Dos modos de operación

Controlado por `kuberay.serving.online` en `k3s/params.yaml`:

```yaml
kuberay:
  serving:
    online: false   # ← cambiar a true para modo online
```

---

### Modo `online: false` (batch — DSL en Spark)

```
Kafka
  │
  ▼
Spark (kafka_main.py)
  ├─ kafka_to_schema_features()   → schema convertido
  ├─ PipelineModel.transform()    → 14 features (DSL completo)
  └─ POST /infer {"data": [[f1,...,f14]]}
          │
          ▼
     Ray Serve
       └─ modelo.predict()   (solo predicción, sin preprocesar)
          │
          ▼
     {"predictions": [...], "probabilities": [...], ...}
          │
          ▼
Spark escribe resultado a Kafka output topic
```

- Ray Serve carga el modelo una vez al arrancar; no vuelve a leer MLflow.
- No se registra webhook en MLflow.

---

### Modo `online: true` (DSL en Ray Serve)

```
Kafka
  │
  ▼
Spark (kafka_main.py)
  └─ kafka_to_schema_features()   → schema convertido (sin DSL)
     └─ POST /infer {"raw": {"timestamp": ..., "src_port": ..., ...}}
             │
             ▼
        Ray Serve
          ├─ NumpyPipelineExecutor.transform_to_vector()  → 14 features
          └─ modelo.predict()
             │
             ▼
        {"predictions": [...], "probabilities": [...], ...}
             │
             ▼
Spark escribe resultado a Kafka output topic
```

- Artifacts (stages.json + config.json) se resuelven al arrancar:
  `MLflow alias → artifact_set_id tag → pyiceberg → pipeline_hash → S3`
- El webhook MLflow triggerea `reconfigure()` → recarga modelo **y** executor sin reiniciar.
- Spark **no carga** el PipelineModel DSL ni lo descarga de S3.

---

## API: ejemplos de llamada al endpoint

El endpoint es `POST /infer`.

### Modo `online: false` — Spark envía features pre-procesadas

Spark ejecutó el DSL completo y manda las 14 features en orden:

```bash
curl -s -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[
      2.0,
      3.0,
      5.0,
      0.0,
      0.6931471805599453,
      0.5,
      0.3333333333333333,
      0.0,
      1.0,
      -1.0,
      0.8660254037844387,
      0.5,
      0.9396926207859084,
      0.3420201433256687
    ]]
  }'
```

Las 14 posiciones siguen el orden de `final_features.features` en `dsl_001.yaml`.

**Respuesta:**
```json
{
  "predictions": [0],
  "probabilities": [[0.92, 0.03, 0.02, 0.01, 0.01, 0.01]],
  "latency_ms": 1.8,
  "model": {
    "variant": "stable",
    "framework": "pytorch",
    "registry": "attack-detection",
    "alias": "champion",
    "version": "7"
  }
}
```

---

### Modo `online: true` — Spark envía el evento con schema correcto

Spark solo hizo `kafka_to_schema_features` (sin DSL). El dict está ya plano y tipado:

```bash
curl -s -X POST http://serving.localhost/infer \
  -H "Content-Type: application/json" \
  -d '{
    "raw": {
      "event_id":          "b3d1c2a0-1234-4f56-89ab-cdef01234567",
      "timestamp":         1735691403,
      "src_port":          12345,
      "dst_port":          80,
      "protocol":          "TCP",
      "packet_count":      10,
      "conn_state":        "SF",
      "bytes_transferred": 1024.0
    }
  }'
```

`timestamp` es epoch seconds (int), igual a como lo produce `kafka_to_schema_features`
en Spark (`to_timestamp().cast("long")`).

**Respuesta (misma estructura):**
```json
{
  "predictions": [2],
  "probabilities": [[0.04, 0.03, 0.88, 0.02, 0.02, 0.01]],
  "latency_ms": 3.2,
  "model": {
    "variant": "stable",
    "framework": "pytorch",
    "registry": "attack-detection",
    "alias": "champion",
    "version": "7"
  }
}
```

### Health check

```bash
curl http://localhost:8000/infer
# {"status": "ok", "route": "/infer", "message": "Use POST with JSON payload."}
```

### Webhook MLflow (solo `online: true`)

```bash
curl -s -X POST http://serving.localhost/infer/webhook \
  -H "Content-Type: application/json" \
  -H "X-Mlflow-Event-Timestamp: 1735691403000" \
  -H "X-Mlflow-Signature: sha256=..." \
  -d '{
    "event_type": "MODEL_VERSION_ALIASED",
    "data": {
      "name": "attack-detection",
      "alias": "champion",
      "version": "8"
    }
  }'
```

---

## Qué hace cada componente

### `kafka_main.py` — dos modos

| Paso | `online: false` | `online: true` |
|---|---|---|
| Leer Kafka | ✅ | ✅ |
| `kafka_to_schema_features` | ✅ | ✅ |
| DSL `PipelineModel.transform` | ✅ | ❌ |
| Descargar artifacts de S3 | ✅ (startup) | ❌ |
| Payload a Ray Serve | `{"data": [[f1…f14]]}` | `{"raw": {schema_dict}}` |
| Escribir a Kafka | ✅ | ✅ |

### `src/dsl/numpy_executor.py` — executor sin Spark

Lee los mismos `stages.json` + `config.json` que usa Spark. Ejecuta el DSL
row-by-row sobre un dict Python. No hay lógica duplicada.

| DSL type | Python impl |
|---|---|
| `cast_transformer` | datetime / float / int cast |
| `log_transformer` | `math.log1p / log / log10 / log2` |
| `concat_transformer` | `sep.join(...)` |
| `string_indexer` | dict lookup |
| `temporal_extractor` | `datetime.hour / isoweekday() / ...` |
| `cyclic_transformer` | `math.sin / cos` |
| `arithmetic_transformer` | add / sub / mul / div / pow / abs / neg |
| `conditional_transformer` | isin / gt / lt / eq / is_null |
| `standard_scaler` | `(x - mean) / std` |
| `minmax_scaler` | `(x - min) / (max - min)` |
| `ratio_transformer` | `a / b` |
| `binning_transformer` | bisect sobre bins |
| `clip_transformer` | `max(lo, min(hi, x))` |
| `fillna_transformer` | fill constante |
| `imputer` | learned fill value |
| `frequency_encoder` | dict lookup + other_key |

### `src/serve/pipeline_loader.py` — resuelve artifacts sin Spark

```
MLflow alias (champion / challenger)
    └─► artifact_set_id tag en la versión del modelo
          └─► pyiceberg: metadata.preprocessing_artifacts → pipeline_hash
                └─► S3: pipelines/{pipeline_hash}/stages.json
                         pipelines/{pipeline_hash}/config.json
                    └─► NumpyPipelineExecutor
```

Equivalencias con `kafka_main.py`:

| `kafka_main.py` | `pipeline_loader.py` |
|---|---|
| `_resolve_artifact_set_id_from_mlflow()` | `_resolve_artifact_set_id()` |
| `_resolve_pipeline_hash_from_metadata()` (Spark SQL) | `_resolve_pipeline_hash()` (pyiceberg directo) |
| `_download_pipeline_from_key()` | `_download_artifacts()` |

### `src/serve/runtime.py` — dispatcher de payloads

- `"raw"` en payload → `executor.transform_to_vector(payload["raw"])` → `[feature_vec]`
- `"data"` en payload → `normalize_payload(payload)` (path existente, sin cambios)
- El modelo se carga una vez al arrancar. En `online: true` el webhook triggerea `reconfigure()`.

---

## Test: dsl_001.yaml con 2026_01_01_001.parquet

`notebooks/test_numpy_pipeline.ipynb` valida que NumPy y Spark producen el mismo resultado:

```
notebooks/out/2026_01_01_001.parquet (3 000 filas)
    ├─► Spark fit + transform → columnas de referencia
    └─► NumpyPipelineExecutor.transform_to_vector() × 3 000
              └─► np.allclose(atol=1e-7, rtol=1e-5)
```

- Columnas categoriales: diferencia = 0.0 (lookup de dict exacto).
- Columnas continuas normalizadas: diferencia < 1e-10.

Latencia del executor NumPy: ~50–200 µs por fila (sin JVM, sin serialización).
