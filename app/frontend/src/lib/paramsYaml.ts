/**
 * Client-side params_training.yaml generator.
 *
 * Produce la misma estructura que el backend _generate_training_params_yaml() emite,
 * para el panel de preview YAML en el Training tab.
 *
 * Nota: spark block y serving/canary ya NO están aquí — pertenecen a
 * params_preprocess.yaml y params_serving.yaml respectivamente.
 *
 * S3 paths use <S3_BUCKET> as a placeholder (bucket resolved server-side).
 */

import { stringify } from 'yaml';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface AdvancedConfig {
  mlflow_tracking_uri: string;
  mlflow_artifact_location: string;
  serving_alias: string;
  canary: boolean;
  canary_alias: string;
  canary_probability: number;
  webhook_public_base_url: string;
  webhook_path: string;
  webhook_name: string;
  webhook_max_timestamp_age_seconds: number;
  spark_read_batch_size: number;
  spark_write_batch_size: number;
  iceberg_warehouse: string;
}

export interface ParamsYamlInput {
  execution_id: string;
  dataset: string;
  dslS3Path: string;          // "s3://..." or placeholder
  tuning: {
    enabled: boolean;
    number_of_trials: number;
  };
  framework: 'xgboost' | 'pytorch';
  model: {
    experiment_name: string;
    registry_model_name: string;
    target: string;
    num_classes: number;
    seed: number;
  };
  sample_fraction_for_tuning: number;
  hyperparams: Record<string, unknown>;
  tuneSettings: Record<string, unknown>;
  advanced: AdvancedConfig;
}

// ─── Generator ────────────────────────────────────────────────────────────────

export function generateParamsYaml(input: ParamsYamlInput): string {
  const {
    execution_id, tuning,
    framework, model, sample_fraction_for_tuning,
    hyperparams, tuneSettings, advanced,
  } = input;

  const hyperparamsBlock: Record<string, unknown> = {
    [framework]: hyperparams,
  };
  if (tuning.enabled && Object.keys(tuneSettings).length > 0) {
    hyperparamsBlock['tuning'] = tuneSettings;
  }

  // params_training.yaml — sin spark block, sin raw_table/dsl_s3_path, sin kuberay.serving/canary
  const params = {
    execution: {
      execution_id,
      tuning: {
        enabled: tuning.enabled,
        number_of_trials: tuning.number_of_trials,
      },
      skip_preprocessing: true,
    },
    model: {
      framework,
      experiment_name: model.experiment_name,
      registry_model_name: model.registry_model_name,
      target: model.target,
      num_classes: model.num_classes,
      tune: tuning.enabled,
      sample_fraction_for_tuning,
      seed: model.seed,
    },
    hyperparams: hyperparamsBlock,
    kuberay: {
      model: {
        tune: tuning.enabled,
        sample_fraction_for_tuning,
        target: model.target,
        num_classes: model.num_classes,
        dsl_count_dim: true,
        input_dim: 14,
        framework,
        seed: model.seed,
        mlflow_tracking_uri: advanced.mlflow_tracking_uri,
        mlflow_experiment_name: model.experiment_name,
        mlflow_artifact_location: advanced.mlflow_artifact_location,
        mlflow_registry_model_name: model.registry_model_name,
      },
    },
    iceberg_tables: {
      warehouse: advanced.iceberg_warehouse,
      metadata: {
        catalog: 'iceberg',
        namespace: 'metadata',
        table: 'preprocessing_artifacts',
        full_name: 'iceberg.metadata.preprocessing_artifacts',
      },
      processed: {
        catalog: 'iceberg',
        namespace: 'processed',
      },
    },
  };

  return stringify(params, { indent: 2 });
}

/** Default AdvancedConfig matching k3s/params.yaml values */
export const DEFAULT_ADVANCED_CONFIG: AdvancedConfig = {
  // Empty means "use backend/Airflow default" (MLFLOW_TRACKING_URI env var).
  mlflow_tracking_uri: '',
  mlflow_artifact_location: 's3://k8s-mlops-platform-bucket/mlflow-artifacts/',
  serving_alias: 'champion',
  canary: false,
  canary_alias: 'challenger',
  canary_probability: 0.1,
  webhook_public_base_url: 'http://model-serving-serve-svc.ray.svc.cluster.local:8000',
  webhook_path: '/infer/webhook',
  webhook_name: 'rayserve-attack-detection-champion-alias',
  webhook_max_timestamp_age_seconds: 300,
  spark_read_batch_size: 512,
  spark_write_batch_size: 100000,
  iceberg_warehouse: 's3://k8s-mlops-platform-bucket/warehouse',
};
