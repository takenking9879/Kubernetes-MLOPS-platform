# K3s GPU Runtime (containerd)

Este documento explica el flujo de los scripts GPU de `k3s` usando `containerd` y cómo manejar la imagen manualmente (one-time o por versión).

## Qué hace cada script

- `k3s/up-k3s-gpu.sh`
  - Levanta `k3s`.
  - Configura runtime NVIDIA para pods GPU (RuntimeClass `nvidia` + device plugin).
  - Verifica que el nodo tenga `nvidia.com/gpu` allocatable.
  - Verifica que tu imagen ya exista en `k3s containerd` (no importa ni guarda).

- `k3s/down-k3s-gpu.sh`
  - Baja `k3s`.
  - Opcionalmente baja dependencias (`docker`, `containerd` host).
  - No desinstala k3s ni borra datos de `/var/lib/rancher/k3s`.

- `k3s/test-k3s-gpu.sh`
  - Crea un pod con tu imagen local en `containerd`.
  - Ejecuta `nvidia-smi`.
  - Falla con diagnóstico si no hay GPU o la imagen no está.

## Flujo normal diario

```bash
sudo bash k3s/up-k3s-gpu.sh
sudo bash k3s/test-k3s-gpu.sh
```

Si quieres apagar todo:

```bash
sudo bash k3s/down-k3s-gpu.sh
```

## Importación manual one-time de imagen a k3s containerd

Ejemplo con tu imagen:

```bash
docker save -o /tmp/ray-train-2.53.0.tar takenking9879/ray-train:2.53.0
sudo k3s ctr -n k8s.io images import /tmp/ray-train-2.53.0.tar
sudo k3s ctr -n k8s.io images ls | grep -E 'takenking9879/ray-train|2.53.0'
```

Notas:

- `k3s ctr` suele registrar la referencia como `docker.io/takenking9879/ray-train:2.53.0`.
- El test usa `imagePullPolicy: Never`, así que la imagen debe existir localmente en `k3s containerd`.

## Imagenes criticas que deben existir en k3s containerd

Estas imagenes suelen estar en Docker Desktop, pero no necesariamente en el containerd de k3s.

- Airflow: `apache/airflow-k8s`
- DSL app: `dsl-app`
- SkyPilot API: `berkeleyskypilot/skypilot`
- Sky runner/controllers: `takenking9879/sky-runner`
- Spark: `k3s-spark-cluster`
- LLMs: `takenking9879/ray-llm`

Importacion sugerida (ajusta tags segun tu version):

```bash
# Desde Docker local -> tar -> k3s containerd
docker save -o /tmp/apache-airflow-k8s.tar apache/airflow-k8s:latest
docker save -o /tmp/dsl-app.tar dsl-app:latest
docker save -o /tmp/berkeleyskypilot-skypilot.tar berkeleyskypilot/skypilot:latest
docker save -o /tmp/takenking9879-sky-runner.tar takenking9879/sky-runner:latest
docker save -o /tmp/k3s-spark-cluster.tar k3s-spark-cluster:latest
docker save -o /tmp/takenking9879-ray-llm.tar takenking9879/ray-llm:latest

sudo k3s ctr -n k8s.io images import /tmp/apache-airflow-k8s.tar
sudo k3s ctr -n k8s.io images import /tmp/dsl-app.tar
sudo k3s ctr -n k8s.io images import /tmp/berkeleyskypilot-skypilot.tar
sudo k3s ctr -n k8s.io images import /tmp/takenking9879-sky-runner.tar
sudo k3s ctr -n k8s.io images import /tmp/k3s-spark-cluster.tar
sudo k3s ctr -n k8s.io images import /tmp/takenking9879-ray-llm.tar

sudo k3s ctr -n k8s.io images ls | grep -E \
'apache/airflow-k8s|dsl-app|berkeleyskypilot/skypilot|takenking9879/sky-runner|k3s-spark-cluster|takenking9879/ray-llm'
```

## Cuando saques una nueva versión de imagen

Ejemplo: nueva tag `2.54.0`.

1. Construir o traer imagen local.

```bash
docker pull takenking9879/ray-train:2.54.0
# o: docker build -t takenking9879/ray-train:2.54.0 .
```

2. Importarla manualmente a `k3s containerd`.

```bash
docker save -o /tmp/ray-train-2.54.0.tar takenking9879/ray-train:2.54.0
sudo k3s ctr -n k8s.io images import /tmp/ray-train-2.54.0.tar
sudo k3s ctr -n k8s.io images ls | grep -E 'takenking9879/ray-train|2.54.0'
```

3. Probar con la nueva versión.

```bash
sudo TEST_POD_IMAGE=takenking9879/ray-train:2.54.0 bash k3s/test-k3s-gpu.sh
```

4. Opcional: actualizar precheck de `up`.

```bash
sudo LOCAL_IMAGE=takenking9879/ray-train:2.54.0 bash k3s/up-k3s-gpu.sh
```

## Limpieza opcional de versiones viejas

Ver imágenes:

```bash
sudo k3s ctr -n k8s.io images ls | grep takenking9879/ray-train
```

Borrar una tag vieja:

```bash
sudo k3s ctr -n k8s.io images rm docker.io/takenking9879/ray-train:2.53.0
```

Si te marca que está en uso, primero elimina pods/workloads que referencien esa imagen.
