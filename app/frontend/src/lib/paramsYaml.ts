/**
 * Client-side params.yaml generator.
 *
 * Produces the exact YAML structure that the backend _generate_params_yaml()
 * emits, for use in the live YAML preview panel.
 *
 * S3 paths use <S3_BUCKET> as a placeholder since the bucket name is
 * resolved server-side.
 */

import { stringify } from 'yaml';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SplitRange {
  start: string;
  end: string;
}

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
  splits: {
    train: SplitRange;
    val: SplitRange;
    test: SplitRange;
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
    execution_id, dataset, dslS3Path, tuning, splits,
    framework, model, sample_fraction_for_tuning,
    hyperparams, tuneSettings, advanced,
  } = input;

  const BUCKET = '<S3_BUCKET>';

  const hyperparamsBlock: Record<string, unknown> = {
    [framework]: hyperparams,
  };
  if (tuning.enabled && Object.keys(tuneSettings).length > 0) {
    hyperparamsBlock['tuning'] = tuneSettings;
  }

  const params = {
    execution: {
      execution_id,
      raw_table: `iceberg.raw.${dataset}`,
      dsl_s3_path: dslS3Path,
      tuning: {
        enabled: tuning.enabled,
        number_of_trials: tuning.number_of_trials,
      },
    },
    splits: {
      train: { start: splits.train.start, end: splits.train.end },
      val:   { start: splits.val.start,   end: splits.val.end },
      test:  { start: splits.test.start,  end: splits.test.end },
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
      serving: {
        alias: advanced.serving_alias,
        canary: advanced.canary,
        webhook_public_base_url: advanced.webhook_public_base_url,
        webhook_path: advanced.webhook_path,
        webhook_name: advanced.webhook_name,
        webhook_max_timestamp_age_seconds: advanced.webhook_max_timestamp_age_seconds,
      },
      canary: {
        alias: advanced.canary_alias,
        canary_probability: advanced.canary_probability,
        initial_replicas: 0,
      },
    },
    spark: {
      app_name: 'spark-preprocessing',
      bucket: BUCKET,
      read_batch_size: advanced.spark_read_batch_size,
      write_batch_size: advanced.spark_write_batch_size,
      num_classes: model.num_classes,
      target: model.target,
      schemas: {
        input_schema: 'kafka_schema_features',
        features_schema: 'schema_features',
        output_schema: 'prediction_schema',
        preprocessed_schema: 'schema_preprocessed',
        full_schema: 'schema_full',
      },
      converters: {
        kafka_to_features: 'kafka_to_schema_features',
      },
      prediction: {
        type: 'ray_serve',
        ray_serve: {
          url: `${advanced.webhook_public_base_url}/infer`,
          batch_size: 256,
          timeout: 30,
          max_retries: 3,
          request_payload_format: 'list',
          response_format: 'json',
          prediction_key: 'predictions',
        },
        columns: {
          id_column: 'event_id',
          prediction_column: 'label',
        },
      },
      output: {
        format: 'json',
        key_column: 'event_id',
        value_columns: ['timestamp', 'event_id', 'properties', 'label'],
      },
      checkpoint: {
        location: `s3a://${BUCKET}/checkpoints/kafka-spark-inference`,
        cleanup_on_exit: false,
      },
      operational: {
        check_kafka_connection: true,
        log_level: 'INFO',
      },
      streaming: {
        online_processing_time: '250 milliseconds',
        starting_offsets: 'latest',
        fail_on_data_loss: false,
      },
    },
    iceberg_tables: {
      warehouse: advanced.iceberg_warehouse,
      raw: {
        catalog: 'iceberg',
        namespace: 'raw',
        table: dataset,
        full_name: `iceberg.raw.${dataset}`,
      },
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
  mlflow_tracking_uri: 'http://my-mlflow',
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
