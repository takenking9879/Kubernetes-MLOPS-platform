/**
 * DAG orchestration types for the ZENTHROSML CANVAS pipeline view.
 *
 * Each OrchestrationNode wraps one pipeline step.
 * Edges propagate artifact IDs (datasetName, preprocessRunId, trainRunId) downstream automatically.
 *
 * MVP: single input per node. Multi-input (MergeNode) is documented in
 * context/dag_multi_input_future.md and stubbed in the palette as "Coming soon".
 */

import type { Node, Edge } from 'reactflow';
import type { ResourceConstraints } from '../api/platformClient';

// ─── Status ───────────────────────────────────────────────────────────────────

export type NodeStatus = 'idle' | 'configured' | 'running' | 'success' | 'error';

// ─── Node kinds ───────────────────────────────────────────────────────────────

export type OrchNodeKind = 'dataset' | 'processing' | 'training' | 'serving';

// Disabled stub kinds — rendered in palette but not functional
export type StubNodeKind = 'merge' | 'decision' | 'model_selector';

// ─── Split config (reused from ProcessingPage) ────────────────────────────────

export interface SplitBound {
  start: string;
  end: string;
}

export interface SplitConfig {
  train: SplitBound;
  val: SplitBound;
  test: SplitBound;
}

// ─── Node data shapes ─────────────────────────────────────────────────────────

export interface DatasetOrchNodeData {
  nodeKind: 'dataset';
  label: string;
  status: NodeStatus;
  errors: string[];
  datasetName: string;
}

export interface ProcessingOrchNodeData {
  nodeKind: 'processing';
  label: string;
  status: NodeStatus;
  errors: string[];
  // propagated from upstream DatasetNode
  datasetName: string;
  dslVersion: number | '';
  executionId: string;
  splits: SplitConfig;
  preprocessRunId: string | null;
  dagRunId: string | null;
}

export interface TrainingOrchNodeData {
  nodeKind: 'training';
  label: string;
  status: NodeStatus;
  errors: string[];
  // propagated from upstream ProcessingNode
  preprocessRunId: string | null;
  // model category: tabular | llm | upload
  modelCategory: 'tabular' | 'llm' | 'upload';
  modelType: 'tabular' | 'llm';
  framework: 'xgboost' | 'pytorch';
  taskType: 'classification' | 'regression';
  pytorchModelType: 'mlp' | 'ssm' | 'bae';
  // model config (POST /api/v2/runs → model field)
  experimentName: string;
  registryModelName: string;
  mlflowTrackingUri: string;
  mlflowArtifactLocation: string;
  target: string;
  numClasses: number;
  seed: number;
  // training flags / sky launch
  useManagedJobs: boolean;   // true = sky jobs launch, false = sky launch --retry-until-up
  useInfiniband: boolean;
  // hyperparams (fixed values when tuning disabled)
  hyperparams: Record<string, number | string | string[]>;
  // tuning
  tuneEnabled: boolean;
  numTrials: number;
  sampleFraction: number;
  tuneSettings: Record<string, number>;
  searchSpaceOverrides: Record<string, { min?: number; max?: number; options?: number[] }>;
  finalTrainOverrides: Record<string, number>;
  // LLM-specific
  llmModelId: string;
  hfToken: string;
  datasetS3: string;
  maxSteps: number;
  useDeepspeed: boolean;
  // upload architecture
  uploadedArchS3: string;
  uploadedArchId: string;
  // resources
  resourceConstraints: ResourceConstraints;
  trainRunId: string | null;
  dagRunId: string | null;
}

export interface ServingOrchNodeData {
  nodeKind: 'serving';
  label: string;
  status: NodeStatus;
  errors: string[];
  // propagated from upstream TrainingNode
  trainRunId: string | null;
  // 'ray_only' = tabular Ray Serve, 'kafka' = kafka inference, 'vllm' = vLLM via SkyPilot
  servingMode: 'ray_only' | 'kafka' | 'vllm' | '';
  // tabular serving config (POST /api/v2/serving-configs)
  alias: string;
  canary: boolean;
  canaryAlias: string;
  canaryProbability: number;
  initialReplicas: number;
  minReplicas: number;
  maxReplicas: number;
  targetQpsPerReplica: number;
  rawSchemaS3Path: string;   // required for kafka mode
  webhookPublicBaseUrl: string;
  webhookPath: string;
  webhookMaxTimestampAge: number;
  // tabular serving GPU resources
  servingResourceConstraints: ResourceConstraints;
  // SkyServe controller config
  serveControllerHighAvailability: boolean;
  serveControllerInfra: string;
  serveControllerCpus: string;
  serveControllerDiskSize: number;
  // vLLM config (used when servingMode === 'vllm', POST /api/v2/jobs/launch)
  hfToken: string;
  llmAdapterS3: string;
  vllmPort: number;
  maxModelLen: number;
  tensorParallelSize: number;
  pipelineParallelSize: number;
  vllmReplicas: number;
  servingRunId: string | null;
  dagRunId: string | null;
}

export type OrchestrationNodeData =
  | DatasetOrchNodeData
  | ProcessingOrchNodeData
  | TrainingOrchNodeData
  | ServingOrchNodeData;

// ─── ReactFlow node/edge types ────────────────────────────────────────────────

export type OrchestrationNode = Node<OrchestrationNodeData>;

export interface OrchestrationEdgeData {
  sourceArtifactType: 'dataset_name' | 'preprocess_run_id' | 'train_run_id';
  propagatedValue: string | null;
}

export type OrchestrationEdge = Edge<OrchestrationEdgeData>;

// ─── Pipeline manifest (exported as JSON for reproducibility) ─────────────────

export interface PipelineManifest {
  manifestVersion: '1.0';
  pipelineName: string;
  pipelineVersion: string;
  createdAt: string;
  nodes: Array<{
    id: string;
    nodeKind: OrchNodeKind;
    position: { x: number; y: number };
    config: OrchestrationNodeData;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    data: OrchestrationEdgeData;
  }>;
}

// ─── Default data factories ────────────────────────────────────────────────────

export function makeDatasetNodeData(overrides?: Partial<DatasetOrchNodeData>): DatasetOrchNodeData {
  return {
    nodeKind: 'dataset',
    label: 'Dataset',
    status: 'idle',
    errors: [],
    datasetName: '',
    ...overrides,
  };
}

export function makeProcessingNodeData(overrides?: Partial<ProcessingOrchNodeData>): ProcessingOrchNodeData {
  return {
    nodeKind: 'processing',
    label: 'Processing',
    status: 'idle',
    errors: [],
    datasetName: '',
    dslVersion: '',
    executionId: '',
    splits: {
      train: { start: '2026-01-01 00:30:00', end: '2026-01-01 13:00:00' },
      val:   { start: '2026-01-01 13:00:00', end: '2026-01-02 05:40:00' },
      test:  { start: '2026-01-02 07:00:00', end: '2026-01-03 03:50:00' },
    },
    preprocessRunId: null,
    dagRunId: null,
    ...overrides,
  };
}

export function makeTrainingNodeData(overrides?: Partial<TrainingOrchNodeData>): TrainingOrchNodeData {
  return {
    nodeKind: 'training',
    label: 'Training',
    status: 'idle',
    errors: [],
    preprocessRunId: null,
    modelCategory: 'tabular',
    modelType: 'tabular',
    framework: 'xgboost',
    taskType: 'classification',
    pytorchModelType: 'mlp',
    experimentName: '',
    registryModelName: '',
    mlflowTrackingUri: '',
    mlflowArtifactLocation: 's3://k8s-mlops-platform-bucket/mlflow-artifacts/',
    target: '',
    numClasses: 2,
    seed: 42,
    useManagedJobs: false,
    useInfiniband: false,
    hyperparams: {},
    tuneEnabled: true,
    numTrials: 3,
    sampleFraction: 0.2,
    tuneSettings: {},
    searchSpaceOverrides: {},
    finalTrainOverrides: {},
    llmModelId: '',
    hfToken: '',
    datasetS3: '',
    maxSteps: 500,
    useDeepspeed: false,
    uploadedArchS3: '',
    uploadedArchId: '',
    resourceConstraints: {
      providers: ['runpod'],
      num_nodes: 1,
      num_gpus_per_node: 1,
      prefer_spot: true,
      job_type: 'tabular',
    },
    trainRunId: null,
    dagRunId: null,
    ...overrides,
  };
}

export function makeServingNodeData(overrides?: Partial<ServingOrchNodeData>): ServingOrchNodeData {
  return {
    nodeKind: 'serving',
    label: 'Serving',
    status: 'idle',
    errors: [],
    trainRunId: null,
    servingMode: 'ray_only',
    alias: 'champion',
    canary: false,
    canaryAlias: 'challenger',
    canaryProbability: 0.1,
    initialReplicas: 1,
    minReplicas: 1,
    maxReplicas: 3,
    targetQpsPerReplica: 10,
    rawSchemaS3Path: '',
    webhookPublicBaseUrl: '',
    webhookPath: '/infer/webhook',
    webhookMaxTimestampAge: 300,
    servingResourceConstraints: { providers: ['runpod'], num_nodes: 1, prefer_spot: false, job_type: 'tabular' },
    serveControllerHighAvailability: false,
    serveControllerInfra: '',
    serveControllerCpus: '4+',
    serveControllerDiskSize: 100,
    hfToken: '',
    llmAdapterS3: '',
    vllmPort: 8000,
    maxModelLen: 4096,
    tensorParallelSize: 1,
    pipelineParallelSize: 1,
    vllmReplicas: 1,
    servingRunId: null,
    dagRunId: null,
    ...overrides,
  };
}
