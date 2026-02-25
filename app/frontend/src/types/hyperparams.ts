/**
 * Hyperparameter type definitions mirroring the Python backend exactly.
 *
 * Three distinct sources:
 *   *_DEFAULTS       → base model defaults (*_PARAMS)
 *   *_SEARCH_SPACE   → Ray Tune search space (SEARCH_SPACE_*)
 *   *_TUNE_SETTINGS  → per-trial execution budget (*_TUNE_SETTINGS)
 *
 * Precedence:
 *   tune=false → *_DEFAULTS for all params
 *   tune=true  → *_TUNE_SETTINGS for training-length, SEARCH_SPACE_* for tuned params,
 *                *_DEFAULTS for non-SEARCH_SPACE params (constant across trials)
 */

// ─── Shared types ─────────────────────────────────────────────────────────────

export type ParamType = 'number' | 'string' | 'array';

export interface ParamMeta {
  type: ParamType;
  defaultValue: number | string | string[];
  /** For string params with a fixed set of valid values */
  options?: string[];
  /** Display hint for numeric inputs */
  min?: number;
  max?: number;
  step?: number;
}

export type SearchSpaceEntry =
  | { type: 'randint'; min: number; max: number }
  | { type: 'choice'; options: (number | string)[] }
  | { type: 'uniform'; min: number; max: number }
  | { type: 'loguniform'; min: number; max: number }
  | { type: 'fixed'; value: number | string | string[] };

// ─── XGBoost ──────────────────────────────────────────────────────────────────

export const XGBOOST_DEFAULTS: Record<string, number | string | string[]> = {
  num_boost_round: 60,
  objective: 'multi:softprob',
  eval_metric: ['mlogloss', 'merror'],
  booster: 'gbtree',
  tree_method: 'hist',
  verbosity: 1,
  eta: 0.3,
  max_depth: 6,
  min_child_weight: 1,
  subsample: 1.0,
  lambda: 1.0,
  alpha: 0.0,
};

export const XGBOOST_PARAM_META: Record<string, ParamMeta> = {
  num_boost_round:  { type: 'number', defaultValue: 60,               min: 1,    step: 1,    options: undefined },
  objective:        { type: 'string', defaultValue: 'multi:softprob', options: ['multi:softprob', 'multi:softmax'] },
  eval_metric:      { type: 'array',  defaultValue: ['mlogloss', 'merror'] },
  booster:          { type: 'string', defaultValue: 'gbtree',         options: ['gbtree', 'gblinear', 'dart'] },
  tree_method:      { type: 'string', defaultValue: 'hist',           options: ['hist', 'approx', 'exact'] },
  verbosity:        { type: 'number', defaultValue: 1,                min: 0,    max: 3,  step: 1 },
  eta:              { type: 'number', defaultValue: 0.3,              min: 0,    max: 1,  step: 0.01 },
  max_depth:        { type: 'number', defaultValue: 6,                min: 1,    max: 20, step: 1 },
  min_child_weight: { type: 'number', defaultValue: 1,                min: 0,    step: 1 },
  subsample:        { type: 'number', defaultValue: 1.0,              min: 0.01, max: 1,  step: 0.01 },
  lambda:           { type: 'number', defaultValue: 1.0,              min: 0,    step: 0.001 },
  alpha:            { type: 'number', defaultValue: 0.0,              min: 0,    step: 0.001 },
};

export const XGBOOST_ALLOWED_KEYS = new Set(Object.keys(XGBOOST_DEFAULTS));

/** Keys that Ray Tune optimizes when tune=true */
export const XGBOOST_SEARCH_SPACE: Record<string, SearchSpaceEntry> = {
  objective:        { type: 'fixed',     value: 'multi:softprob' },
  eval_metric:      { type: 'fixed',     value: ['mlogloss', 'merror'] },
  max_depth:        { type: 'randint',   min: 3, max: 11 },
  min_child_weight: { type: 'choice',    options: [1, 2, 3] },
  subsample:        { type: 'uniform',   min: 0.5, max: 1.0 },
  eta:              { type: 'loguniform', min: 1e-4, max: 0.3 },
  lambda:           { type: 'loguniform', min: 1e-3, max: 10.0 },
  alpha:            { type: 'loguniform', min: 1e-3, max: 10.0 },
};

/** Per-trial execution budget (scheduler settings). Overrides num_boost_round from *_PARAMS. */
export const XGBOOST_TUNE_SETTINGS_DEFAULTS: Record<string, number> = {
  num_boost_round: 10,
  grace_period: 5,
  reduction_factor: 2,
};

export const XGBOOST_TUNE_SETTINGS_META: Record<string, ParamMeta> = {
  num_boost_round:  { type: 'number', defaultValue: 10, min: 1, step: 1 },
  grace_period:     { type: 'number', defaultValue: 5,  min: 1, step: 1 },
  reduction_factor: { type: 'number', defaultValue: 2,  min: 2, step: 1 },
};

// ─── PyTorch ──────────────────────────────────────────────────────────────────

export const PYTORCH_DEFAULTS: Record<string, number | string | string[]> = {
  batch_size:   256,
  max_epochs:   50,
  lr:           1e-3,
  weight_decay: 0.0,
};

export const PYTORCH_PARAM_META: Record<string, ParamMeta> = {
  batch_size:   { type: 'number', defaultValue: 256,  min: 1,   step: 1 },
  max_epochs:   { type: 'number', defaultValue: 50,   min: 1,   step: 1 },
  lr:           { type: 'number', defaultValue: 1e-3, min: 1e-7, max: 1.0, step: 1e-5 },
  weight_decay: { type: 'number', defaultValue: 0.0,  min: 0,   step: 1e-6 },
};

export const PYTORCH_ALLOWED_KEYS = new Set(Object.keys(PYTORCH_DEFAULTS));

/** Keys that Ray Tune optimizes when tune=true */
export const PYTORCH_SEARCH_SPACE: Record<string, SearchSpaceEntry> = {
  batch_size:   { type: 'choice',    options: [64, 128, 256] },
  lr:           { type: 'loguniform', min: 1e-5, max: 0.1 },
  weight_decay: { type: 'loguniform', min: 1e-6, max: 1e-2 },
};

/** Per-trial execution budget (ASHA scheduler settings). Overrides max_epochs from *_PARAMS. */
export const PYTORCH_TUNE_SETTINGS_DEFAULTS: Record<string, number> = {
  max_epochs: 10,
  grace_period: 5,
  reduction_factor: 2,
};

export const PYTORCH_TUNE_SETTINGS_META: Record<string, ParamMeta> = {
  max_epochs:       { type: 'number', defaultValue: 10, min: 1, step: 1 },
  grace_period:     { type: 'number', defaultValue: 5,  min: 1, step: 1 },
  reduction_factor: { type: 'number', defaultValue: 2,  min: 2, step: 1 },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

export function getDefaults(framework: 'xgboost' | 'pytorch') {
  return framework === 'xgboost' ? { ...XGBOOST_DEFAULTS } : { ...PYTORCH_DEFAULTS };
}

export function getTuneSettingsDefaults(framework: 'xgboost' | 'pytorch') {
  return framework === 'xgboost'
    ? { ...XGBOOST_TUNE_SETTINGS_DEFAULTS }
    : { ...PYTORCH_TUNE_SETTINGS_DEFAULTS };
}

export function getAllowedKeys(framework: 'xgboost' | 'pytorch') {
  return framework === 'xgboost' ? XGBOOST_ALLOWED_KEYS : PYTORCH_ALLOWED_KEYS;
}

export function getSearchSpace(framework: 'xgboost' | 'pytorch') {
  return framework === 'xgboost' ? XGBOOST_SEARCH_SPACE : PYTORCH_SEARCH_SPACE;
}

export function getParamMeta(framework: 'xgboost' | 'pytorch') {
  return framework === 'xgboost' ? XGBOOST_PARAM_META : PYTORCH_PARAM_META;
}

export function getTuneSettingsMeta(framework: 'xgboost' | 'pytorch') {
  return framework === 'xgboost' ? XGBOOST_TUNE_SETTINGS_META : PYTORCH_TUNE_SETTINGS_META;
}

/** Keys that are NOT in the search space — applied as constants across all trials */
export function getNonTunableKeys(framework: 'xgboost' | 'pytorch'): string[] {
  const searchSpace = getSearchSpace(framework);
  const allowed = getAllowedKeys(framework);
  const tuneSettings = getTuneSettingsDefaults(framework);
  // Non-tunable = allowed keys that are NOT in search space and NOT in tune settings
  return [...allowed].filter(
    (k) => !(k in searchSpace) && !(k in tuneSettings)
  );
}
