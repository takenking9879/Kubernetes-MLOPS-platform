"""
NumPy executor for fitted DSL pipelines.

Executes a fitted DSL pipeline on a single row (dict) using pure Python/NumPy.
Zero Spark dependency. Designed for low-latency, row-by-row inference in Ray Serve.

The executor reads the same stages.json + config.json that Spark uses, so the
DSL YAML remains the single source of truth — no logic is duplicated.

Usage:
    executor = NumpyPipelineExecutor.from_dir("/path/to/pipeline/artifacts")
    row = {"timestamp": 1735691403, "src_port": 12345, "protocol": "TCP", ...}
    features = executor.transform_to_vector(row)  # ordered List[float]
"""

import json
import math
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NumpyPipelineExecutor:
    """
    Execute a fitted DSL pipeline on a single row dict using pure Python.

    Loads the same artifacts (stages.json + config.json) that Spark uses at
    inference time. No Spark, no Ray Data in the critical path.

    Input:  dict  — column name → Python scalar (str, int, float, datetime)
    Output: dict  — enriched with all intermediate and final columns
            OR
            List[float] — ordered feature vector (via transform_to_vector)
    """

    def __init__(self, stages: List[Dict], final_features: List[str]):
        self._stages = stages
        self._final_features = final_features  # ordered output column names

        self._runners = {
            "cast_transformer":        self._cast,
            "log_transformer":         self._log,
            "concat_transformer":      self._concat,
            "string_indexer":          self._string_indexer,
            "temporal_extractor":      self._temporal,
            "cyclic_transformer":      self._cyclic,
            "arithmetic_transformer":  self._arithmetic,
            "conditional_transformer": self._conditional,
            "standard_scaler":         self._standard_scaler,
            "minmax_scaler":           self._minmax_scaler,
            "ratio_transformer":       self._ratio,
            "binning_transformer":     self._binning,
            "clip_transformer":        self._clip,
            "fillna_transformer":      self._fillna,
            "imputer":                 self._imputer,
            "frequency_encoder":       self._frequency_encoder,
        }

    # ── Factory ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_dir(cls, artifact_dir: str) -> "NumpyPipelineExecutor":
        """
        Load from an artifact directory containing stages.json and config.json.
        Mirrors PipelineModel.load() without any Spark dependency.
        """
        import os
        stages_path = os.path.join(artifact_dir, "stages.json")
        config_path = os.path.join(artifact_dir, "config.json")

        with open(stages_path) as f:
            stages = json.load(f)

        final_features: List[str] = []
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            ff = config.get("final_features", {})
            if isinstance(ff, dict):
                final_features = list(ff.get("features", []))

        logger.info(
            "NumpyPipelineExecutor loaded: %d stages, %d output features",
            len(stages), len(final_features),
        )
        return cls(stages, final_features)

    # ── Public API ───────────────────────────────────────────────────────────────

    @property
    def final_features(self) -> List[str]:
        return list(self._final_features)

    def transform(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all pipeline stages to one row dict. Returns enriched row."""
        row = dict(row)  # never mutate the caller's dict
        for stage in self._stages:
            stype = stage["type"]
            runner = self._runners.get(stype)
            if runner is None:
                raise ValueError(
                    f"Unknown DSL stage type: {stype!r} (stage name: {stage.get('name')})"
                )
            row = runner(row, stage)
        return row

    def transform_to_vector(self, row: Dict[str, Any]) -> List[float]:
        """
        Transform a row and return an ordered feature vector matching final_features.
        This is what gets sent to the ML model.
        """
        if not self._final_features:
            raise RuntimeError(
                "final_features is empty — check config.json in the artifact dir."
            )
        enriched = self.transform(row)
        missing = [c for c in self._final_features if c not in enriched]
        if missing:
            raise KeyError(
                f"Missing output columns after pipeline: {missing}. "
                f"Available: {sorted(enriched)}"
            )
        return [float(enriched[col]) for col in self._final_features]

    # ── Helpers ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _ins(stage: Dict) -> List[str]:
        v = stage["inputs"]
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _outs(stage: Dict) -> List[str]:
        v = stage["outputs"]
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _get(row: Dict, col: str) -> Any:
        """Fetch column value with case-insensitive fallback."""
        if col in row:
            return row[col]
        col_l = col.lower().strip()
        for k, v in row.items():
            if k.lower().strip() == col_l:
                return v
        raise KeyError(
            f"Column {col!r} not found in row. Available: {sorted(row)}"
        )

    # ── Stage Runners ────────────────────────────────────────────────────────────

    def _cast(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        t = s["params"]["target_type"]
        val = self._get(row, col_in)

        if t == "timestamp":
            if isinstance(val, datetime):
                pass  # already good (pandas Timestamp is a datetime subclass)
            elif isinstance(val, (int, float)):
                # Epoch seconds — from Kafka converter: to_timestamp().cast("long")
                val = datetime.utcfromtimestamp(float(val))
            else:
                val = _parse_ts_str(str(val))
        elif t in ("double", "float"):
            val = float(val)
        elif t in ("integer", "int", "long"):
            val = int(float(val))
        elif t == "string":
            val = str(val)

        row[col_out] = val
        return row

    def _log(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        v  = float(self._get(row, col_in))
        lt = s["params"]["log_type"]

        if lt == "log1p":
            result = math.log1p(v)
        elif lt == "log":
            result = math.log(v) if v > 0 else 0.0
        elif lt == "log10":
            result = math.log10(v) if v > 0 else 0.0
        elif lt == "log2":
            result = math.log2(v) if v > 0 else 0.0
        else:
            raise ValueError(f"Unknown log_type: {lt!r}")

        row[col_out] = result
        return row

    def _concat(self, row: Dict, s: Dict) -> Dict:
        sep     = s["params"].get("separator", "")
        col_out = self._outs(s)[0]
        parts   = [str(self._get(row, c)) for c in self._ins(s)]
        row[col_out] = sep.join(parts)
        return row

    def _string_indexer(self, row: Dict, s: Dict) -> Dict:
        # learned_params: {"mappings": {col: {"mapping": {str: int}, "num_labels": int}}}
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        handle  = s["params"].get("handle_invalid", "keep")
        mapping = s["learned_params"]["mappings"][col_in]["mapping"]
        val     = str(self._get(row, col_in))

        if val in mapping:
            idx: Optional[float] = float(mapping[val])
        else:
            if handle == "keep":
                idx = -1.0
            elif handle == "skip":
                idx = None
            else:
                raise ValueError(
                    f"Unseen label {val!r} in column {col_in!r} (handle_invalid='error')"
                )

        row[col_out] = idx
        return row

    def _temporal(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        comp    = s["params"]["component"]
        dt      = self._get(row, col_in)

        if not isinstance(dt, datetime):
            raise TypeError(
                f"Temporal extraction requires a datetime, got {type(dt).__name__} "
                f"for column {col_in!r}. Did you forget cast_transformer first?"
            )

        if comp == "hour":
            result = dt.hour
        elif comp == "minute":
            result = dt.minute
        elif comp == "second":
            result = dt.second
        elif comp == "dayofweek":
            # Match Spark's dayofweek(): 1=Sunday, 2=Monday, ..., 7=Saturday
            # Python isoweekday(): 1=Monday, ..., 7=Sunday
            # Conversion: spark = (iso % 7) + 1
            #   Monday:   iso=1 → (1%7)+1=2  ✓
            #   Saturday: iso=6 → (6%7)+1=7  ✓
            #   Sunday:   iso=7 → (7%7)+1=1  ✓
            result = (dt.isoweekday() % 7) + 1
        elif comp == "dayofmonth":
            result = dt.day
        elif comp == "month":
            result = dt.month
        elif comp == "year":
            result = dt.year
        elif comp == "quarter":
            result = (dt.month - 1) // 3 + 1
        elif comp == "weekofyear":
            result = dt.isocalendar()[1]
        else:
            raise ValueError(f"Unknown temporal component: {comp!r}")

        row[col_out] = result
        return row

    def _cyclic(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        fn      = s["params"]["function"]
        period  = float(s["params"]["period"])
        v       = float(self._get(row, col_in))
        angle   = 2.0 * math.pi * v / period

        row[col_out] = math.sin(angle) if fn == "sin" else math.cos(angle)
        return row

    def _arithmetic(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        op      = s["params"]["operation"]
        value   = s["params"].get("value", 0)
        v       = float(self._get(row, col_in))

        ops = {
            "add":      v + value,
            "subtract": v - value,
            "multiply": v * value,
            "divide":   v / value if value != 0 else 0.0,
            "power":    v ** value,
            "absolute": abs(v),
            "negate":   -v,
        }
        if op not in ops:
            raise ValueError(f"Unknown arithmetic operation: {op!r}")

        row[col_out] = ops[op]
        return row

    def _conditional(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        cond    = s["params"]["condition"]
        true_v  = s["params"].get("true_value", 1)
        false_v = s["params"].get("false_value", 0)
        val     = self._get(row, col_in)

        if cond == "isin":
            match = val in s["params"]["values"]
        elif cond == "greater_than":
            match = val > s["params"]["value"]
        elif cond == "less_than":
            match = val < s["params"]["value"]
        elif cond == "equals":
            match = val == s["params"]["value"]
        elif cond == "is_null":
            match = val is None
        else:
            raise ValueError(f"Unknown condition: {cond!r}")

        row[col_out] = true_v if match else false_v
        return row

    def _standard_scaler(self, row: Dict, s: Dict) -> Dict:
        # learned_params: {"stats": {col: {"mean": float, "std": float}}}
        stats = s["learned_params"]["stats"]
        for col_in, col_out in zip(self._ins(s), self._outs(s)):
            v   = float(self._get(row, col_in))
            col_stats = stats[col_in]
            m   = col_stats["mean"]
            sd  = col_stats["std"] or 1.0  # guard against std=0
            row[col_out] = (v - m) / sd
        return row

    def _minmax_scaler(self, row: Dict, s: Dict) -> Dict:
        # learned_params: {"stats": {col: {"min", "max", "range_min", "range_max"}}}
        stats = s["learned_params"]["stats"]
        for col_in, col_out in zip(self._ins(s), self._outs(s)):
            v  = float(self._get(row, col_in))
            cs = stats[col_in]
            mn, mx = cs["min"], cs["max"]
            lo, hi = cs["range_min"], cs["range_max"]
            if mx == mn:
                row[col_out] = lo
            else:
                row[col_out] = lo + (v - mn) / (mx - mn) * (hi - lo)
        return row

    def _ratio(self, row: Dict, s: Dict) -> Dict:
        cols    = self._ins(s)
        col_out = self._outs(s)[0]
        num     = float(self._get(row, cols[0]))
        den     = float(self._get(row, cols[1]))
        row[col_out] = num / den if den != 0 else 0.0
        return row

    def _binning(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        bins    = s["params"]["bins"]
        labels  = s["params"].get("labels")
        v       = float(self._get(row, col_in))

        bucket = len(bins) - 2  # last bin by default
        for i in range(1, len(bins)):
            if v < bins[i]:
                bucket = i - 1
                break

        row[col_out] = labels[bucket] if labels else float(bucket)
        return row

    def _clip(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        lo      = s["params"].get("min_value")
        hi      = s["params"].get("max_value")
        v       = float(self._get(row, col_in))

        if lo is not None:
            v = max(v, float(lo))
        if hi is not None:
            v = min(v, float(hi))

        row[col_out] = v
        return row

    def _fillna(self, row: Dict, s: Dict) -> Dict:
        col_in  = self._ins(s)[0]
        col_out = self._outs(s)[0]
        val     = self._get(row, col_in)

        if val is None or (isinstance(val, float) and math.isnan(val)):
            strategy = s["params"].get("strategy", "constant")
            if strategy == "constant":
                val = s["params"].get("fill_value", 0)
            else:
                # forward_fill not meaningful for a single row
                val = 0

        row[col_out] = val
        return row

    def _imputer(self, row: Dict, s: Dict) -> Dict:
        # learned_params: {"imputation_values": {col: value}}
        fill_vals = s["learned_params"]["imputation_values"]
        for col_in, col_out in zip(self._ins(s), self._outs(s)):
            val = self._get(row, col_in)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = fill_vals.get(col_in, 0)
            row[col_out] = val
        return row

    def _frequency_encoder(self, row: Dict, s: Dict) -> Dict:
        # learned_params: {"mappings": {col: {"mapping": {str: float}, "other_key": str, ...}}}
        all_mappings = s["learned_params"]["mappings"]
        handle       = s["params"].get("handle_invalid", "keep")

        for col_in, col_out in zip(self._ins(s), self._outs(s)):
            col_data  = all_mappings[col_in]
            mapping   = col_data["mapping"]
            other_key = col_data.get("other_key", "__OTHER__")
            encoding  = col_data.get("encoding", "freq")
            default   = float(s["params"].get(
                "fill_value", 0.0 if encoding == "freq" else -1.0
            ))
            val = str(self._get(row, col_in))

            if val in mapping:
                row[col_out] = float(mapping[val])
            elif other_key in mapping:
                row[col_out] = float(mapping[other_key])
            else:
                if handle == "keep":
                    row[col_out] = default
                elif handle == "skip":
                    row[col_out] = None
                else:
                    raise ValueError(
                        f"Unseen value {val!r} in frequency_encoder for column {col_in!r}"
                    )
        return row


# ── Module-level helpers ─────────────────────────────────────────────────────

def _parse_ts_str(s: str) -> datetime:
    """Parse an ISO-8601 timestamp string to a naive datetime (matching Spark)."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]
    # Strip timezone offset so the result is naive (matches Spark's TimestampType)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Fallback — handles ISO strings with offset; strip it
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=None)
