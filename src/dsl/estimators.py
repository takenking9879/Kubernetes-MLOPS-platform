"""
Concrete Estimator implementations.

All estimators in this module learn parameters from data during fit()
and return FittedTransformer instances that can transform new data.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from typing import Dict, Any, Tuple, List
from pyspark.sql.types import NumericType, StringType

from .base import Estimator, FittedTransformer


# =============================================================================
# STRING INDEXER (Categorical Encoding)
# =============================================================================

class StringIndexerEstimator(Estimator):
    """
    Learn a mapping from categorical strings to integer indices.
    Similar to Spark ML's StringIndexer but outputs a regular column.
    Supports N→N: each input column gets its own vocabulary.
    """

    def get_type(self) -> str:
        return "string_indexer"

    def fit(self, df: DataFrame) -> 'StringIndexerFittedTransformer':
        """Learn the vocabulary from each input column."""
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"StringIndexerEstimator '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )

        mappings = {}
        for input_col in self.inputs:
            values = (
                df.select(input_col)
                  .distinct()
                  .rdd
                  .map(lambda r: r[0])
                  .collect()
            )
            sorted_values = sorted([v for v in values if v is not None])
            mapping = {v: i for i, v in enumerate(sorted_values)}
            mappings[input_col] = {
                "mapping": mapping,
                "num_labels": len(mapping)
            }

        learned_params = {"mappings": mappings}

        self._fitted = True

        return StringIndexerFittedTransformer(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            params=self.params,
            learned_params=learned_params
        )


class StringIndexerFittedTransformer(FittedTransformer):
    """Applies learned string-to-index mapping. Supports N→N."""

    def get_type(self) -> str:
        return "string_indexer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"StringIndexerFittedTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        handle_invalid = self.params.get("handle_invalid", "keep")
        all_mappings = self.learned_params["mappings"]

        for in_col, out_col in zip(self.inputs, self.outputs):
            col_data = all_mappings[in_col]
            mapping = col_data["mapping"]

            if not mapping:
                raise ValueError(
                    f"StringIndexerFittedTransformer has empty mapping for column '{in_col}'. "
                    f"Cannot transform data."
                )

            mapping_items = []
            for k, v in mapping.items():
                mapping_items.append(F.lit(k))
                mapping_items.append(F.lit(v))
            mapping_expr = F.create_map(*mapping_items)

            if handle_invalid == "keep":
                df = df.withColumn(
                    out_col,
                    F.coalesce(mapping_expr[F.col(in_col)], F.lit(-1))
                )
            elif handle_invalid == "skip":
                known_values = list(mapping.keys())
                df = df.filter(F.col(in_col).isin(known_values)).withColumn(
                    out_col,
                    mapping_expr[F.col(in_col)]
                )
            elif handle_invalid == "error":
                unknown_check = F.when(
                    mapping_expr[F.col(in_col)].isNull(), F.lit(True)
                ).otherwise(F.lit(False))
                if df.filter(unknown_check).limit(1).count() > 0:
                    raise ValueError(
                        f"Unknown category found in column '{in_col}' while handle_invalid='error'"
                    )
                df = df.withColumn(out_col, mapping_expr[F.col(in_col)])
            else:
                raise ValueError(f"Unknown handle_invalid: {handle_invalid}")

        return df

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'StringIndexerFittedTransformer':
        """Reconstruct from saved configuration."""
        return cls(
            name=config["name"],
            inputs=config["inputs"],
            outputs=config["outputs"],
            params=config["params"],
            learned_params=config["learned_params"]
        )


# =============================================================================
# STANDARD SCALER (Numerical Normalization)
# =============================================================================

class StandardScalerEstimator(Estimator):
    """
    Learn mean and standard deviation from numerical columns.
    Applies z-score normalization: (x - mean) / std
    """
    
    def get_type(self) -> str:
        return "standard_scaler"
    
    def fit(self, df: DataFrame) -> 'StandardScalerFittedTransformer':
        """Learn mean and std for each input column."""
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"StandardScalerEstimator '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        with_mean = self.params.get("with_mean", True)
        with_std = self.params.get("with_std", True)
        
        stats = {}
        
        for col in self.inputs:
            # Compute statistics
            row = df.select(
                F.mean(col).alias("mean"),
                F.stddev(col).alias("std")
            ).collect()[0]
            
            stats[col] = {
                "mean": float(row["mean"]) if with_mean else 0.0,
                "std": float(row["std"] or 1.0) if with_std else 1.0
            }
        
        learned_params = {
            "stats": stats
        }
        
        self._fitted = True
        
        return StandardScalerFittedTransformer(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            params=self.params,
            learned_params=learned_params
        )


class StandardScalerFittedTransformer(FittedTransformer):
    """Applies learned standardization to numerical columns."""
    
    def get_type(self) -> str:
        return "standard_scaler"
    
    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"StandardScalerFittedTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        stats = self.learned_params["stats"]
        
        # Apply standardization to each column
        for input_col, output_col in zip(self.inputs, self.outputs):
            if input_col not in stats:
                raise ValueError(f"Column {input_col} not found in learned stats")
            
            col_stats = stats[input_col]
            mean = col_stats["mean"]
            std = col_stats["std"]
            
            # Handle case where std is 0
            if std == 0:
                std = 1.0
            
            df = df.withColumn(
                output_col,
                (F.col(input_col) - F.lit(mean)) / F.lit(std)
            )
        
        return df
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'StandardScalerFittedTransformer':
        """Reconstruct from saved configuration."""
        return cls(
            name=config["name"],
            inputs=config["inputs"],
            outputs=config["outputs"],
            params=config["params"],
            learned_params=config["learned_params"]
        )


# =============================================================================
# MIN-MAX SCALER (Alternative Numerical Normalization)
# =============================================================================

class MinMaxScalerEstimator(Estimator):
    """
    Learn min and max from numerical columns.
    Applies min-max normalization: (x - min) / (max - min)
    """
    
    def get_type(self) -> str:
        return "minmax_scaler"
    
    def fit(self, df: DataFrame) -> 'MinMaxScalerFittedTransformer':
        """Learn min and max for each input column."""
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"MinMaxScalerEstimator '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        feature_range = self.params.get("feature_range", (0, 1))
        
        stats = {}
        
        for col in self.inputs:
            # Compute min and max
            row = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max")
            ).collect()[0]
            
            stats[col] = {
                "min": float(row["min"]),
                "max": float(row["max"]),
                "range_min": feature_range[0],
                "range_max": feature_range[1]
            }
        
        learned_params = {
            "stats": stats
        }
        
        self._fitted = True
        
        return MinMaxScalerFittedTransformer(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            params=self.params,
            learned_params=learned_params
        )


class MinMaxScalerFittedTransformer(FittedTransformer):
    """Applies learned min-max scaling to numerical columns."""
    
    def get_type(self) -> str:
        return "minmax_scaler"
    
    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"MinMaxScalerFittedTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        stats = self.learned_params["stats"]
        
        for input_col, output_col in zip(self.inputs, self.outputs):
            if input_col not in stats:
                raise ValueError(f"Column {input_col} not found in learned stats")
            
            col_stats = stats[input_col]
            data_min = col_stats["min"]
            data_max = col_stats["max"]
            range_min = col_stats["range_min"]
            range_max = col_stats["range_max"]
            
            # Handle case where max == min
            if data_max == data_min:
                df = df.withColumn(output_col, F.lit(range_min))
            else:
                # Scale to [range_min, range_max]
                scaled = (F.col(input_col) - F.lit(data_min)) / F.lit(data_max - data_min)
                scaled = scaled * F.lit(range_max - range_min) + F.lit(range_min)
                df = df.withColumn(output_col, scaled)
        
        return df
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'MinMaxScalerFittedTransformer':
        """Reconstruct from saved configuration."""
        return cls(
            name=config["name"],
            inputs=config["inputs"],
            outputs=config["outputs"],
            params=config["params"],
            learned_params=config["learned_params"]
        )

# =============================================================================
# IMPUTER (Fill missing values with learned statistics)
# =============================================================================

class ImputerEstimator(Estimator):
    """
    Learn imputation values (mean, median, mode) from data.
    """
    def get_type(self) -> str:
        return "imputer"

    def fit(self, df: DataFrame) -> 'ImputerFittedTransformer':
        """Learn imputation values for each column."""
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"ImputerEstimator '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        strategy = self.params.get("strategy", "mean")
        schema = df.schema
        imputation_values = {}

        for col in self.inputs:
            if col not in schema:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            dtype = schema[col].dataType
            is_numeric = isinstance(dtype, NumericType)
            is_categorical = isinstance(dtype, StringType)

            # -------------------------------
            # Strategy vs type validation
            # -------------------------------
            if strategy in ("mean", "median") and not is_numeric:
                raise ValueError(
                    f"Imputer strategy '{strategy}' only supports numeric columns. "
                    f"Column '{col}' is {dtype}"
                )

            if strategy == "mode" and not (is_numeric or is_categorical):
                raise ValueError(
                    f"Imputer strategy 'mode' not supported for column '{col}' of type {dtype}"
                )
            # -------------------------------
            # Compute imputation value
            # -------------------------------
            if strategy == "mean":
                value = df.select(F.mean(col)).collect()[0][0]

            elif strategy == "median":
                value = df.approxQuantile(col, [0.5], 0.01)[0]

            elif strategy == "mode":
                value = (
                    df.groupBy(col)
                      .count()
                      .orderBy(F.desc("count"))
                      .select(col)
                      .first()[0]
                )

            elif strategy == "constant":
                value = self.params.get("fill_value")
                if value is None:
                    raise ValueError("Constant strategy requires 'fill_value' param")

            else:
                raise ValueError(f"Unknown imputer strategy: {strategy}")

            imputation_values[col] = value

        learned_params = {
            "imputation_values": imputation_values
        }

        self._fitted = True

        return ImputerFittedTransformer(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            params=self.params,
            learned_params=learned_params
        )

class ImputerFittedTransformer(FittedTransformer):
    """Applies learned imputation to columns with missing values."""
    
    def get_type(self) -> str:
        return "imputer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"ImputerFittedTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        imputation_values = self.learned_params["imputation_values"]
        
        for input_col, output_col in zip(self.inputs, self.outputs):
            if input_col not in imputation_values:
                raise ValueError(f"Column {input_col} not found in learned imputation values")
            
            fill_value = imputation_values[input_col]
            
            df = df.withColumn(
                output_col,
                F.coalesce(F.col(input_col), F.lit(fill_value))
            )
        
        return df
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ImputerFittedTransformer':
        """Reconstruct from saved configuration."""
        return cls(
            name=config["name"],
            inputs=config["inputs"],
            outputs=config["outputs"],
            params=config["params"],
            learned_params=config["learned_params"]
        )
    
# =============================================================================
# FREQUENCY ENCODER
# =============================================================================

class FrequencyEncoderEstimator(Estimator):
    """
    Learn frequency / count encoding for categorical columns.
    Supports N→N: each input column gets its own frequency mapping.

    Params supported (via self.params):
      - encoding: "count" | "freq"   (default: "freq")
      - smoothing: float (Laplace alpha, default 0.0 -> no smoothing)
      - top_k: int or None (keep top_k categories, rest -> other)
      - min_count: int or None (categories with count < min_count -> other)
      - other_key: value to use as the 'other' category label (default "__OTHER__")
      - create_map_threshold: int (<= this size we use F.create_map, default 1000)
      - broadcast_threshold: int (<= this size we broadcast join, default 10000)
      - fill_value: numeric value to use for unknowns (default 0.0 for freq, or -1 for counts)
      - handle_invalid: "keep" | "skip" | "error" (default "keep")
    """
    def get_type(self) -> str:
        return "frequency_encoder"

    def fit(self, df: DataFrame) -> 'FrequencyFittedTransformer':
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"FrequencyEncoderEstimator '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )

        encoding = self.params.get("encoding", "freq")
        smoothing = float(self.params.get("smoothing", 0.0))
        top_k = self.params.get("top_k", None)
        min_count = self.params.get("min_count", None)
        other_key = self.params.get("other_key", "__OTHER__")
        create_map_threshold = int(self.params.get("create_map_threshold", 1000))
        broadcast_threshold = int(self.params.get("broadcast_threshold", 10000))

        mappings = {}

        for input_col in self.inputs:
            counts_df = df.groupBy(input_col).count()
            collected = counts_df.rdd.map(lambda r: (r[0], int(r[1]))).collect()

            total_count = int(df.count()) if encoding == "freq" or smoothing > 0.0 else None

            sorted_by_count = sorted(collected, key=lambda kv: kv[1], reverse=True)

            if min_count is not None:
                keep_set = {k for k, c in sorted_by_count if c >= int(min_count)}
            else:
                keep_set = {k for k, _ in sorted_by_count}

            if top_k is not None:
                top_items = [k for k, _ in sorted_by_count[:int(top_k)]]
                keep_set = set(top_items)

            other_count = 0
            final_counts: Dict[Any, int] = {}
            for k, c in sorted_by_count:
                if k in keep_set:
                    final_counts[k] = c
                else:
                    other_count += c

            if other_count > 0:
                final_counts[other_key] = other_count

            mapping_values: Dict[Any, float] = {}
            V = len(final_counts)
            if encoding == "count":
                for k, c in final_counts.items():
                    mapping_values[k] = float(c + smoothing) if smoothing > 0.0 else float(c)
            elif encoding == "freq":
                if total_count is None:
                    total_count = int(df.count())
                denom = (total_count + smoothing * V) if smoothing > 0.0 else float(total_count)
                for k, c in final_counts.items():
                    num = (c + smoothing) if smoothing > 0.0 else float(c)
                    mapping_values[k] = float(num) / float(denom) if denom != 0 else 0.0
            else:
                raise ValueError(f"Unknown encoding: {encoding}")

            mappings[input_col] = {
                "mapping": mapping_values,
                "other_key": other_key,
                "encoding": encoding,
                "smoothing": smoothing,
                "cardinality": len(mapping_values),
                "total_count": total_count,
            }

        learned_params = {
            "mappings": mappings,
            "create_map_threshold": create_map_threshold,
            "broadcast_threshold": broadcast_threshold,
            "params_snapshot": {k: self.params.get(k) for k in ["top_k", "min_count", "handle_invalid"]}
        }

        self._fitted = True

        return FrequencyFittedTransformer(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            params=self.params,
            learned_params=learned_params
        )


class FrequencyFittedTransformer(FittedTransformer):
    """
    Applies learned frequency / count mapping. Supports N→N.
    The transformer chooses between:
      - F.create_map mapping expression (fast for small mappings)
      - broadcast(join) against a small static DataFrame (recommended for larger mappings)
    """

    def get_type(self) -> str:
        return "frequency_encoder"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"FrequencyFittedTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        spark = SparkSession.builder.getOrCreate()
        all_mappings = self.learned_params["mappings"]
        create_map_threshold = int(self.learned_params.get("create_map_threshold", 1000))
        broadcast_threshold = int(self.learned_params.get("broadcast_threshold", 10000))
        handle_invalid = self.params.get("handle_invalid", "keep")

        for in_col, out_col in zip(self.inputs, self.outputs):
            col_data = all_mappings[in_col]
            mapping: Dict[Any, float] = col_data["mapping"]
            encoding = col_data.get("encoding", "freq")
            default_fill = float(self.params.get(
                "fill_value", 0.0 if encoding == "freq" else -1.0
            ))

            map_size = len(mapping)

            if map_size <= create_map_threshold:
                lit_seq = []
                for k, v in mapping.items():
                    lit_seq.append(F.lit(k))
                    lit_seq.append(F.lit(v))
                mapping_expr = F.create_map(*lit_seq)

                if handle_invalid == "keep":
                    df = df.withColumn(
                        out_col,
                        F.coalesce(mapping_expr[F.col(in_col)], F.lit(default_fill))
                    )
                elif handle_invalid == "skip":
                    known_values = list(mapping.keys())
                    df = df.filter(F.col(in_col).isin(known_values)).withColumn(
                        out_col, mapping_expr[F.col(in_col)]
                    )
                elif handle_invalid == "error":
                    df = df.withColumn(out_col, mapping_expr[F.col(in_col)])
                else:
                    raise ValueError(f"Unknown handle_invalid: {handle_invalid}")

            else:
                mapping_items: List[Tuple[Any, float]] = list(mapping.items())
                key_col_name = f"{in_col}__freq_key"
                val_col_name = f"{in_col}__freq_val"
                map_df = spark.createDataFrame(
                    mapping_items, schema=[key_col_name, val_col_name]
                )

                do_broadcast = map_size <= broadcast_threshold
                join_cond = F.col(in_col) == F.col(key_col_name)

                if do_broadcast:
                    joined = df.join(F.broadcast(map_df), join_cond, how="left")
                else:
                    joined = df.join(map_df, join_cond, how="left")

                if handle_invalid == "keep":
                    df = joined.withColumn(
                        out_col,
                        F.coalesce(F.col(val_col_name), F.lit(default_fill))
                    ).drop(key_col_name, val_col_name)
                elif handle_invalid == "skip":
                    df = joined.filter(
                        F.col(val_col_name).isNotNull()
                    ).withColumn(
                        out_col, F.col(val_col_name)
                    ).drop(key_col_name, val_col_name)
                elif handle_invalid == "error":
                    df = joined.withColumn(
                        out_col, F.col(val_col_name)
                    ).drop(key_col_name, val_col_name)
                else:
                    raise ValueError(f"Unknown handle_invalid: {handle_invalid}")

        return df

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'FrequencyFittedTransformer':
        return cls(
            name=config["name"],
            inputs=config["inputs"],
            outputs=config["outputs"],
            params=config["params"],
            learned_params=config["learned_params"]
        )