/**
 * dagStore — Zustand store for the ZENTHROSML CANVAS orchestration DAG.
 *
 * Responsibilities:
 *   - Nodes and edges of the orchestration graph
 *   - Artifact ID propagation across edges (datasetName, preprocessRunId, trainRunId)
 *   - Manual run ID assignment (reuse existing runs without re-execution)
 *   - Run-node and run-up-to-here logic using existing API calls
 *   - Pipeline manifest export/import
 *   - Persisted to localStorage
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { XYPosition } from 'reactflow';
import {
  type OrchestrationNode,
  type OrchestrationEdge,
  type OrchestrationNodeData,
  type OrchestrationEdgeData,
  type PipelineManifest,
  type NodeStatus,
  type DatasetOrchNodeData,
  type ProcessingOrchNodeData,
  type TrainingOrchNodeData,
  type ServingOrchNodeData,
  makeDatasetNodeData,
  makeProcessingNodeData,
  makeTrainingNodeData,
  makeServingNodeData,
} from '../types/dag';
import {
  submitIngest,
  submitProcessingRun,
  getProcessingRunStatus,
  submitRun,
  launchJob,
  getJobStatus,
  triggerServingDeploy,
  getServingDeployStatus,
  type ResourceConstraints,
} from '../api/platformClient';
import {
  getDefaultsForTask,
  getTuneSettingsDefaults,
  getSearchSpace,
} from '../types/hyperparams';

// ─── Poll helper ──────────────────────────────────────────────────────────────

const POLL_INTERVAL = 8000;
const POLL_TIMEOUT  = 30 * 60 * 1000; // 30 min max

async function pollUntilDone(
  checkFn: () => Promise<{ state: string }>,
  onUpdate?: (state: string) => void,
): Promise<'success' | 'error'> {
  const deadline = Date.now() + POLL_TIMEOUT;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL));
    const result = await checkFn();
    const s = result.state?.toLowerCase() ?? '';
    onUpdate?.(s);
    if (s === 'success') return 'success';
    if (s === 'failed' || s === 'error') return 'error';
  }
  return 'error';
}

// ─── ID helper ───────────────────────────────────────────────────────────────

function nextId(kind: string, existingIds: string[]) {
  const prefix = `${kind}-`;
  let maxSuffix = 0;

  for (const id of existingIds) {
    if (!id.startsWith(prefix)) continue;
    const suffix = Number(id.slice(prefix.length));
    if (Number.isFinite(suffix) && suffix > maxSuffix) {
      maxSuffix = suffix;
    }
  }

  return `${kind}-${maxSuffix + 1}`;
}

// ─── State shape ──────────────────────────────────────────────────────────────

interface DagState {
  nodes: OrchestrationNode[];
  edges: OrchestrationEdge[];
  selectedNodeId: string | null;
  pipelineName: string;
  pipelineVersion: string;

  // ── Node management ──
  addNode: (kind: OrchestrationNodeData['nodeKind'], position?: XYPosition) => void;
  updateNodeData: (id: string, updater: (prev: OrchestrationNodeData) => OrchestrationNodeData) => void;
  deleteNode: (id: string) => void;
  setNodes: (nodes: OrchestrationNode[]) => void;

  // ── Edge management ──
  addEdge: (source: string, target: string) => void;
  removeEdge: (id: string) => void;
  setEdges: (edges: OrchestrationEdge[]) => void;

  // ── Selection ──
  selectNode: (id: string | null) => void;

  // ── Artifact propagation ──
  propagateArtifacts: () => void;

  // ── Run ID assignment (reuse existing runs) ──
  assignRunId: (nodeId: string, runId: string) => void;

  // ── Execution ──
  runNode: (nodeId: string) => Promise<void>;
  runUpTo: (nodeId: string) => Promise<void>;

  // ── Manifest ──
  exportManifest: () => PipelineManifest;
  importManifest: (manifest: PipelineManifest) => void;

  // ── Init ──
  initDefaultPipeline: () => void;
  reset: () => void;
}

// ─── Helper: update node status/data ─────────────────────────────────────────

function applyStatusToNode(
  nodes: OrchestrationNode[],
  id: string,
  status: NodeStatus,
  extra?: Partial<OrchestrationNodeData>,
): OrchestrationNode[] {
  return nodes.map((n) =>
    n.id === id ? { ...n, data: { ...n.data, status, ...extra } as OrchestrationNodeData } : n,
  );
}

// ─── Store ────────────────────────────────────────────────────────────────────

const INITIAL_STATE: Pick<DagState, 'nodes' | 'edges' | 'selectedNodeId' | 'pipelineName' | 'pipelineVersion'> = {
  nodes: [],
  edges: [],
  selectedNodeId: null,
  pipelineName: 'my-pipeline',
  pipelineVersion: '1.0',
};

export const useDagStore = create<DagState>()(
  devtools(
    persist(
      (set, get) => ({
        ...INITIAL_STATE,

        addNode: (kind, position = { x: 200, y: 200 }) => {
          const id = nextId(kind, get().nodes.map((n) => n.id));
          let data: OrchestrationNodeData;
          switch (kind) {
            case 'dataset':    data = makeDatasetNodeData();    break;
            case 'processing': data = makeProcessingNodeData(); break;
            case 'training':   data = makeTrainingNodeData();   break;
            case 'serving':    data = makeServingNodeData();    break;
          }
          const node: OrchestrationNode = { id, type: kind, position, data };
          set((s) => ({ nodes: [...s.nodes, node] }));
        },

        updateNodeData: (id, updater) => {
          set((s) => ({
            nodes: s.nodes.map((n) => n.id === id ? { ...n, data: updater(n.data) } : n),
          }));
          get().propagateArtifacts();
        },

        deleteNode: (id) => {
          set((s) => ({
            nodes: s.nodes.filter((n) => n.id !== id),
            edges: s.edges.filter((e) => e.source !== id && e.target !== id),
            selectedNodeId: s.selectedNodeId === id ? null : s.selectedNodeId,
          }));
        },

        setNodes: (nodes) => set({ nodes }),

        addEdge: (source, target) => {
          const { nodes, edges } = get();
          const src = nodes.find((n) => n.id === source);
          const tgt = nodes.find((n) => n.id === target);
          if (!src || !tgt) return;

          const srcKind = src.data.nodeKind;
          const artifactType: OrchestrationEdgeData['sourceArtifactType'] =
            srcKind === 'dataset'    ? 'dataset_name'      :
            srcKind === 'processing' ? 'preprocess_run_id' :
            'train_run_id';

          const propagatedValue =
            srcKind === 'dataset'    ? (src.data as DatasetOrchNodeData).datasetName        || null :
            srcKind === 'processing' ? (src.data as ProcessingOrchNodeData).preprocessRunId || null :
            srcKind === 'training'   ? (src.data as TrainingOrchNodeData).trainRunId         || null :
            null;

          const edgeId = `e-${source}-${target}`;
          if (edges.some((e) => e.id === edgeId)) return;

          const edge: OrchestrationEdge = {
            id: edgeId,
            source,
            target,
            type: 'fiber',
            data: { sourceArtifactType: artifactType, propagatedValue },
          };

          set((s) => ({ edges: [...s.edges, edge] }));
          get().propagateArtifacts();
        },

        removeEdge: (id) => set((s) => ({ edges: s.edges.filter((e) => e.id !== id) })),

        setEdges: (edges) => set({ edges }),

        selectNode: (id) => set({ selectedNodeId: id }),

        propagateArtifacts: () => {
          const { nodes, edges } = get();
          let updated = [...nodes];

          for (const edge of edges) {
            const srcNode = updated.find((n) => n.id === edge.source);
            const tgtIdx  = updated.findIndex((n) => n.id === edge.target);
            if (!srcNode || tgtIdx === -1) continue;

            const src  = srcNode.data;
            const tgt  = updated[tgtIdx]!.data;
            let changed = false;

            if (src.nodeKind === 'dataset' && tgt.nodeKind === 'processing') {
              const v = (src as DatasetOrchNodeData).datasetName;
              if (v && (tgt as ProcessingOrchNodeData).datasetName !== v) {
                updated[tgtIdx] = { ...updated[tgtIdx]!, data: { ...tgt, datasetName: v } as OrchestrationNodeData };
                changed = true;
              }
            }

            if (src.nodeKind === 'processing' && tgt.nodeKind === 'training') {
              const v = (src as ProcessingOrchNodeData).preprocessRunId;
              if (v && (tgt as TrainingOrchNodeData).preprocessRunId !== v) {
                updated[tgtIdx] = { ...updated[tgtIdx]!, data: { ...tgt, preprocessRunId: v } as OrchestrationNodeData };
                changed = true;
              }
            }

            if (src.nodeKind === 'training' && tgt.nodeKind === 'serving') {
              const v = (src as TrainingOrchNodeData).trainRunId;
              if (v && (tgt as ServingOrchNodeData).trainRunId !== v) {
                updated[tgtIdx] = { ...updated[tgtIdx]!, data: { ...tgt, trainRunId: v } as OrchestrationNodeData };
                changed = true;
              }
            }

            // Update edge propagatedValue
            if (changed) {
              const newVal =
                src.nodeKind === 'dataset'    ? (src as DatasetOrchNodeData).datasetName           :
                src.nodeKind === 'processing' ? (src as ProcessingOrchNodeData).preprocessRunId    :
                src.nodeKind === 'training'   ? (src as TrainingOrchNodeData).trainRunId            :
                null;
              const edgeIdx = get().edges.findIndex((e) => e.id === edge.id);
              if (edgeIdx !== -1) {
                const updEdges = [...get().edges];
                updEdges[edgeIdx] = { ...updEdges[edgeIdx]!, data: { ...edge.data!, propagatedValue: newVal } };
                set({ edges: updEdges });
              }
            }
          }

          set({ nodes: updated });
        },

        assignRunId: (nodeId, runId) => {
          const node = get().nodes.find((n) => n.id === nodeId);
          if (!node) return;

          let updatedData: OrchestrationNodeData;
          switch (node.data.nodeKind) {
            case 'processing':
              updatedData = { ...node.data, preprocessRunId: runId, status: 'success' };
              break;
            case 'training':
              updatedData = { ...node.data, trainRunId: runId, status: 'success' };
              break;
            case 'dataset':
              updatedData = { ...node.data, datasetName: runId, status: 'success' };
              break;
            case 'serving':
              updatedData = { ...node.data, servingRunId: runId, status: 'success' };
              break;
          }

          set((s) => ({
            nodes: s.nodes.map((n) => n.id === nodeId ? { ...n, data: updatedData } : n),
          }));
          get().propagateArtifacts();
        },

        runNode: async (nodeId) => {
          const node = get().nodes.find((n) => n.id === nodeId);
          if (!node) return;

          const setStatus = (status: NodeStatus, extra?: Partial<OrchestrationNodeData>) => {
            set((s) => ({ nodes: applyStatusToNode(s.nodes, nodeId, status, extra) }));
          };

          setStatus('running');

          try {
            const data = node.data;

            if (data.nodeKind === 'dataset') {
              const d = data as DatasetOrchNodeData;
              if (!d.datasetName) throw new Error('Dataset name required');
              const result = await submitIngest(d.datasetName);
              const finalStatus = await pollUntilDone(
                () => Promise.resolve({ state: 'success' }), // ingest polling TBD
              );
              void result;
              setStatus(finalStatus === 'success' ? 'success' : 'error');
            }

            else if (data.nodeKind === 'processing') {
              const d = data as ProcessingOrchNodeData;
              if (!d.datasetName)  throw new Error('Dataset name required');
              if (!d.dslVersion)   throw new Error('DSL version required');

              const resp = await submitProcessingRun({
                dataset: d.datasetName,
                dsl_version: d.dslVersion as number,
                execution_id: d.executionId || undefined,
                splits: {
                  train: d.splits.train,
                  val:   d.splits.val,
                  test:  d.splits.test,
                },
              });

              setStatus('running', { dagRunId: resp.dag_run_id } as Partial<ProcessingOrchNodeData>);

              const finalStatus = await pollUntilDone(() =>
                getProcessingRunStatus(resp.dag_run_id),
              );

              if (finalStatus === 'success') {
                setStatus('success', {
                  preprocessRunId: resp.preprocess_run_id,
                  dagRunId: resp.dag_run_id,
                } as Partial<ProcessingOrchNodeData>);
                get().propagateArtifacts();
              } else {
                setStatus('error');
              }
            }

            else if (data.nodeKind === 'training') {
              const d = data as TrainingOrchNodeData;
              if (!d.preprocessRunId) throw new Error('preprocessRunId required — run or assign Processing node first');

              if (d.modelCategory !== 'llm') {
                // Tabular: POST /api/v2/runs
                const baseHyperparams = d.hyperparams && Object.keys(d.hyperparams).length > 0
                  ? d.hyperparams
                  : getDefaultsForTask(d.framework, d.taskType);
                const baseTuneSettings = d.tuneSettings && Object.keys(d.tuneSettings).length > 0
                  ? d.tuneSettings
                  : getTuneSettingsDefaults(d.framework);

                const resolvedSearchSpace = d.tuneEnabled ? (() => {
                  const baseSpace = getSearchSpace(d.framework);
                  const resolved: Record<string, { type: string; options?: number[]; min?: number; max?: number; value?: unknown }> = {};
                  for (const [k, entry] of Object.entries(baseSpace)) {
                    const ov = d.searchSpaceOverrides[k];
                    type SSVal = { type: string; options?: number[]; min?: number; max?: number; value?: unknown };
                    if (!ov) {
                      resolved[k] = entry as SSVal;
                    } else if (entry.type === 'choice' && ov.options && ov.options.length > 0) {
                      resolved[k] = { type: 'choice', options: ov.options };
                    } else if (entry.type !== 'choice' && (ov.min !== undefined || ov.max !== undefined)) {
                      resolved[k] = {
                        type: entry.type,
                        min: ov.min ?? (entry as { min: number }).min,
                        max: ov.max ?? (entry as { max: number }).max,
                      };
                    } else {
                      resolved[k] = entry as SSVal;
                    }
                  }
                  return resolved;
                })() : undefined;

                const resolvedConstraints: ResourceConstraints = {
                  ...d.resourceConstraints,
                  job_type: 'tabular',
                };

                const useGpu = (resolvedConstraints.providers ?? []).some(p =>
                  ['runpod', 'vast', 'aws', 'gcp', 'azure'].includes(p),
                );
                const resolvedModelType =
                  d.modelCategory === 'upload' ? 'user_uploaded'
                  : d.framework === 'pytorch'  ? d.pytorchModelType
                  : 'xgboost';

                const resp = await submitRun({
                  preprocess_run_id: d.preprocessRunId,
                  framework: d.framework,
                  use_gpu: useGpu,
                  use_managed_jobs: d.useManagedJobs,
                  resource_constraints: resolvedConstraints,
                  model: {
                    experiment_name:          d.experimentName  || `dag_${d.framework}`,
                    registry_model_name:      d.registryModelName || d.framework,
                    mlflow_tracking_uri:      d.mlflowTrackingUri || undefined,
                    mlflow_artifact_location: d.mlflowArtifactLocation || undefined,
                    target:                   d.target || undefined,
                    num_classes:              d.numClasses || undefined,
                    seed:                     d.seed,
                    task_type:                d.taskType,
                    model_type:               resolvedModelType,
                  },
                  tuning: { enabled: d.tuneEnabled, number_of_trials: d.numTrials },
                  sample_fraction_for_tuning: d.sampleFraction,
                  hyperparams: d.tuneEnabled
                    ? (Object.keys(d.finalTrainOverrides).length > 0 ? d.finalTrainOverrides : undefined)
                    : baseHyperparams,
                  tune_settings: d.tuneEnabled ? baseTuneSettings : {},
                  search_space: resolvedSearchSpace,
                  materialize_datasets: d.materializeDatasets,
                });

                setStatus('running', { dagRunId: resp.dag_run_id } as Partial<TrainingOrchNodeData>);

                const finalStatus = await pollUntilDone(() =>
                  getJobStatus(resp.dag_run_id),
                );

                if (finalStatus === 'success') {
                  setStatus('success', {
                    trainRunId: resp.train_run_id,
                    dagRunId: resp.dag_run_id,
                  } as Partial<TrainingOrchNodeData>);
                  get().propagateArtifacts();
                } else {
                  setStatus('error');
                }
              } else {
                // LLM: POST /api/v2/jobs/launch
                const resp = await launchJob({
                  job_type: 'training',
                  model: { model_type: 'llm', model_id: d.llmModelId || undefined },
                  resource_constraints: { ...d.resourceConstraints, job_type: 'llm' },
                  training: {
                    dataset_s3_path: d.datasetS3 || undefined,
                    hf_token:        d.hfToken   || undefined,
                    max_steps:       d.maxSteps,
                    use_deepspeed:   d.useDeepspeed,
                    deepspeed_stage: d.useDeepspeed ? 3 : 1,
                    num_nodes:       d.resourceConstraints.num_nodes ?? 1,
                  },
                });

                const trainDagRunId = resp.job_ids['training'] ?? '';
                setStatus('running', { dagRunId: trainDagRunId } as Partial<TrainingOrchNodeData>);

                const finalStatus = await pollUntilDone(() => getJobStatus(trainDagRunId));

                if (finalStatus === 'success') {
                  setStatus('success', {
                    trainRunId: trainDagRunId,
                    dagRunId: trainDagRunId,
                  } as Partial<TrainingOrchNodeData>);
                  get().propagateArtifacts();
                } else {
                  setStatus('error');
                }
              }
            }

            else if (data.nodeKind === 'serving') {
              const d = data as ServingOrchNodeData;
              if (!d.trainRunId) throw new Error('trainRunId required — run or assign Training node first');

              if (d.servingMode === 'vllm') {
                // vLLM serving via SkyPilot jobs launch
                const resp = await launchJob({
                  job_type: 'serving',
                  model: { model_type: 'llm' },
                  resource_constraints: { ...d.servingResourceConstraints, job_type: 'llm' },
                  serving: {
                    hf_token:               d.hfToken || undefined,
                    llm_adapter_s3:         d.llmAdapterS3 || undefined,
                    vllm_port:              d.vllmPort,
                    max_model_len:          d.maxModelLen,
                    tensor_parallel_size:   d.tensorParallelSize,
                    pipeline_parallel_size: d.pipelineParallelSize,
                    replicas:               d.vllmReplicas,
                  },
                });

                const serveDagRunId = resp.job_ids['serving'] ?? '';
                setStatus('running', { dagRunId: serveDagRunId } as Partial<ServingOrchNodeData>);

                const finalStatus = await pollUntilDone(() => getJobStatus(serveDagRunId));
                setStatus(finalStatus === 'success' ? 'success' : 'error');
              } else {
                // Tabular / kafka: POST /api/v2/serving-configs → deploy
                if (!d.servingRunId) throw new Error('servingRunId required — create serving config first');

                const resp = await triggerServingDeploy(d.servingRunId);
                setStatus('running', { dagRunId: resp.dag_run_id } as Partial<ServingOrchNodeData>);

                const finalStatus = await pollUntilDone(() =>
                  getServingDeployStatus(d.servingRunId!, resp.dag_run_id, resp.dag_id),
                );
                setStatus(finalStatus === 'success' ? 'success' : 'error');
              }
            }

          } catch (err) {
            set((s) => ({
              nodes: applyStatusToNode(s.nodes, nodeId, 'error', {
                errors: [err instanceof Error ? err.message : String(err)],
              } as Partial<OrchestrationNodeData>),
            }));
          }
        },

        runUpTo: async (nodeId) => {
          // Resolve upstream dependency chain (BFS from nodeId toward sources)
          const { nodes, edges } = get();

          const getUpstreamIds = (id: string): string[] => {
            return edges.filter((e) => e.target === id).map((e) => e.source);
          };

          // Topological order from sources to nodeId
          const order: string[] = [];
          const visited = new Set<string>();
          const visit = (id: string) => {
            if (visited.has(id)) return;
            visited.add(id);
            for (const upId of getUpstreamIds(id)) visit(upId);
            order.push(id);
          };
          visit(nodeId);

          // Run each node in order if not already successful
          for (const id of order) {
            const n = get().nodes.find((nd) => nd.id === id);
            if (!n) continue;
            if (n.data.status === 'success') continue;
            await get().runNode(id);
            const after = get().nodes.find((nd) => nd.id === id);
            if (after?.data.status !== 'success') break; // stop on failure
          }
          void nodes; // keep reference used above
        },

        exportManifest: (): PipelineManifest => {
          const { nodes, edges, pipelineName, pipelineVersion } = get();
          return {
            manifestVersion: '1.0',
            pipelineName,
            pipelineVersion,
            createdAt: new Date().toISOString(),
            nodes: nodes.map((n) => ({
              id: n.id,
              nodeKind: n.data.nodeKind,
              position: n.position,
              config: n.data,
            })),
            edges: edges.map((e) => ({
              id: e.id,
              source: e.source,
              target: e.target,
              data: e.data ?? { sourceArtifactType: 'dataset_name', propagatedValue: null },
            })),
          };
        },

        importManifest: (manifest) => {
          const nodes: OrchestrationNode[] = manifest.nodes.map((n) => ({
            id: n.id,
            type: n.nodeKind,
            position: n.position,
            data: n.config,
          }));
          const edges: OrchestrationEdge[] = manifest.edges.map((e) => ({
            id: e.id,
            source: e.source,
            target: e.target,
            type: 'fiber',
            data: e.data,
          }));
          set({ nodes, edges, pipelineName: manifest.pipelineName, pipelineVersion: manifest.pipelineVersion });
        },

        initDefaultPipeline: () => {
          const { nodes } = get();
          if (nodes.length > 0) return; // don't overwrite existing state

          const datasetId    = 'dataset-1';
          const processingId = 'processing-1';
          const trainingId   = 'training-1';

          const defaultNodes: OrchestrationNode[] = [
            {
              id: datasetId,
              type: 'dataset',
              position: { x: 120, y: 220 },
              data: makeDatasetNodeData({ label: 'Dataset' }),
            },
            {
              id: processingId,
              type: 'processing',
              position: { x: 420, y: 220 },
              data: makeProcessingNodeData({ label: 'Processing' }),
            },
            {
              id: trainingId,
              type: 'training',
              position: { x: 720, y: 220 },
              data: makeTrainingNodeData({ label: 'Training' }),
            },
          ];

          const defaultEdges: OrchestrationEdge[] = [
            {
              id: `e-${datasetId}-${processingId}`,
              source: datasetId,
              target: processingId,
              type: 'fiber',
              data: { sourceArtifactType: 'dataset_name', propagatedValue: null },
            },
            {
              id: `e-${processingId}-${trainingId}`,
              source: processingId,
              target: trainingId,
              type: 'fiber',
              data: { sourceArtifactType: 'preprocess_run_id', propagatedValue: null },
            },
          ];

          set({ nodes: defaultNodes, edges: defaultEdges });
        },

        reset: () => set(INITIAL_STATE),
      }),
      {
        name: 'mlops-dag-store',
        version: 2,
        migrate: (_old, _v) => {
          // Version bump: drop old state — new node data shapes are incompatible.
          // initDefaultPipeline() will re-seed on next render.
          return { nodes: [], edges: [], pipelineName: 'my-pipeline', pipelineVersion: '1.0' };
        },
        partialize: (s) => ({
          nodes: s.nodes,
          edges: s.edges,
          pipelineName: s.pipelineName,
          pipelineVersion: s.pipelineVersion,
        }),
      },
    ),
    { name: 'DagStore' },
  ),
);
