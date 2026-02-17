"""
Concrete Transformer implementations.

All transformers in this module are deterministic and do not learn from data.
They implement the Transformer interface from base.py.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DoubleType, IntegerType, StringType, LongType
import math
from typing import Dict, Any

from .base import Transformer


def _normalize_column_name(name: str) -> str:
    return str(name).lstrip("\ufeff").strip().strip('"').strip("'").strip("`").strip().casefold()


def _resolve_input_column(df: DataFrame, requested: str, stage_name: str) -> str:
    if requested in df.columns:
        return requested

    normalized_requested = _normalize_column_name(requested)
    normalized_map: dict[str, str] = {}
    collisions: set[str] = set()

    for current in df.columns:
        norm = _normalize_column_name(current)
        if norm in normalized_map and normalized_map[norm] != current:
            collisions.add(norm)
        else:
            normalized_map[norm] = current

    if normalized_requested in normalized_map and normalized_requested not in collisions:
        return normalized_map[normalized_requested]

    raise ValueError(
        f"{stage_name}: input column '{requested}' not found. Available columns: {df.columns[:30]}"
    )


class CastTransformer(Transformer):
    """Cast a column to a different data type."""

    def get_type(self) -> str:
        return "cast_transformer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"CastTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        target_type = self.params.get("target_type", "string")

        type_mapping = {
            "timestamp": TimestampType(),
            "double": DoubleType(),
            "integer": IntegerType(),
            "string": StringType(),
            "LongType": LongType()
        }

        spark_type = type_mapping.get(target_type, StringType())
        for in_col, out_col in zip(self.inputs, self.outputs):
            resolved_col = _resolve_input_column(df, in_col, f"CastTransformer '{self.name}'")
            df = df.withColumn(out_col, F.col(resolved_col).cast(spark_type))
        return df


class TemporalExtractor(Transformer):
    """Extract temporal components from timestamp columns."""

    def get_type(self) -> str:
        return "temporal_extractor"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"TemporalExtractor '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        component = self.params.get("component", "hour")

        extractors = {
            "hour": F.hour, "minute": F.minute, "second": F.second,
            "dayofweek": F.dayofweek, "dayofmonth": F.dayofmonth,
            "month": F.month, "year": F.year, "quarter": F.quarter,
            "weekofyear": F.weekofyear,
        }

        extractor = extractors.get(component)
        if extractor is None:
            raise ValueError(f"Unknown temporal component: {component}")

        for in_col, out_col in zip(self.inputs, self.outputs):
            df = df.withColumn(out_col, extractor(in_col))
        return df


class ArithmeticTransformer(Transformer):
    """Apply arithmetic operations to columns."""

    def get_type(self) -> str:
        return "arithmetic_transformer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"ArithmeticTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        operation = self.params.get("operation", "add")

        if operation == "add":
            value = self.params.get("value", 0)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, F.col(in_col) + F.lit(value))
        elif operation == "subtract":
            value = self.params.get("value", 0)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, F.col(in_col) - F.lit(value))
        elif operation == "multiply":
            value = self.params.get("value", 1)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, F.col(in_col) * F.lit(value))
        elif operation == "divide":
            value = self.params.get("value", 1)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, F.col(in_col) / F.lit(value))
        elif operation == "power":
            value = self.params.get("value", 2)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, F.pow(F.col(in_col), F.lit(value)))
        elif operation == "absolute":
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, F.abs(F.col(in_col)))
        elif operation == "negate":
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, -F.col(in_col))
        else:
            raise ValueError(f"Unknown operation: {operation}")
        return df


class ConditionalTransformer(Transformer):
    """Apply conditional logic to create new columns."""

    def get_type(self) -> str:
        return "conditional_transformer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"ConditionalTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        condition = self.params.get("condition", "isin")
        true_value = self.params.get("true_value", 1)
        false_value = self.params.get("false_value", 0)

        if condition == "isin":
            values = self.params.get("values", [])
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(
                    out_col,
                    F.when(F.col(in_col).isin(values), true_value).otherwise(false_value)
                )
        elif condition == "greater_than":
            threshold = self.params.get("threshold", 0)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(
                    out_col,
                    F.when(F.col(in_col) > threshold, true_value).otherwise(false_value)
                )
        elif condition == "less_than":
            threshold = self.params.get("threshold", 0)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(
                    out_col,
                    F.when(F.col(in_col) < threshold, true_value).otherwise(false_value)
                )
        elif condition == "equals":
            value = self.params.get("value")
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(
                    out_col,
                    F.when(F.col(in_col) == value, true_value).otherwise(false_value)
                )
        elif condition == "is_null":
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(
                    out_col,
                    F.when(F.col(in_col).isNull(), true_value).otherwise(false_value)
                )
        else:
            raise ValueError(f"Unknown condition: {condition}")
        return df


class CyclicTransformer(Transformer):
    """Apply cyclic encoding (sin/cos) for periodic features."""

    def get_type(self) -> str:
        return "cyclic_transformer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"CyclicTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        function = self.params.get("function", "sin")
        period = self.params.get("period", 24.0)

        trig_fn = {"sin": F.sin, "cos": F.cos}.get(function)
        if trig_fn is None:
            raise ValueError(f"Unknown function: {function}. Use 'sin' or 'cos'.")

        for in_col, out_col in zip(self.inputs, self.outputs):
            angle = 2 * math.pi * F.col(in_col) / period
            df = df.withColumn(out_col, trig_fn(angle))
        return df


class LogTransformer(Transformer):
    """Apply logarithmic transformations."""

    def get_type(self) -> str:
        return "log_transformer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"LogTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        log_type = self.params.get("log_type", "log1p")

        log_fns = {
            "log1p": lambda c: F.log1p(F.col(c)),
            "log": lambda c: F.log(F.col(c)),
            "log10": lambda c: F.log10(F.col(c)),
            "log2": lambda c: F.log2(F.col(c)),
        }

        log_fn = log_fns.get(log_type)
        if log_fn is None:
            raise ValueError(f"Unknown log_type: {log_type}")

        for in_col, out_col in zip(self.inputs, self.outputs):
            df = df.withColumn(out_col, log_fn(in_col))
        return df


class ConcatTransformer(Transformer):
    """Concatenate multiple string columns."""
    
    def get_type(self) -> str:
        return "concat_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        separator = self.params.get("separator", "_")
        output_col = self.outputs[0]
        
        # Concatenate all input columns with separator
        return df.withColumn(
            output_col,
            F.concat_ws(separator, *[F.col(c) for c in self.inputs])
        )


class RatioTransformer(Transformer):
    """Calculate ratio between two columns."""
    
    def get_type(self) -> str:
        return "ratio_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != 2:
            raise ValueError("RatioTransformer requires exactly 2 input columns")
        
        numerator = self.inputs[0]
        denominator = self.inputs[1]
        output_col = self.outputs[0]
        default_value = self.params.get("default_value", 0.0)
        
        # Handle division by zero
        return df.withColumn(
            output_col,
            F.when(
                F.col(denominator) != 0,
                F.col(numerator) / F.col(denominator)
            ).otherwise(default_value)
        )


class BinningTransformer(Transformer):
    """Bin continuous values into discrete buckets."""

    def get_type(self) -> str:
        return "binning_transformer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"BinningTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        bins = self.params.get("bins", [])
        labels = self.params.get("labels", None)

        if not bins:
            raise ValueError("Bins parameter is required")

        default_label = labels[-1] if labels and len(labels) > len(bins) - 1 else len(bins) - 1

        for in_col, out_col in zip(self.inputs, self.outputs):
            expr = None
            for i in range(len(bins) - 1):
                condition = (F.col(in_col) >= bins[i]) & (F.col(in_col) < bins[i + 1])
                label = labels[i] if labels and i < len(labels) else i
                if expr is None:
                    expr = F.when(condition, label)
                else:
                    expr = expr.when(condition, label)
            expr = expr.otherwise(default_label)
            df = df.withColumn(out_col, expr)
        return df


class ClipTransformer(Transformer):
    """Clip values to a specified range."""

    def get_type(self) -> str:
        return "clip_transformer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"ClipTransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        min_value = self.params.get("min_value")
        max_value = self.params.get("max_value")

        for in_col, out_col in zip(self.inputs, self.outputs):
            expr = F.col(in_col)
            if min_value is not None:
                expr = F.when(expr < min_value, min_value).otherwise(expr)
            if max_value is not None:
                expr = F.when(expr > max_value, max_value).otherwise(expr)
            df = df.withColumn(out_col, expr)
        return df


class FillNATransformer(Transformer):
    """Fill missing values with a constant or strategy."""

    def get_type(self) -> str:
        return "fillna_transformer"

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"FillNATransformer '{self.name}': inputs ({len(self.inputs)}) "
                f"and outputs ({len(self.outputs)}) must have the same length"
            )
        strategy = self.params.get("strategy", "constant")

        if strategy == "constant":
            value = self.params.get("value", 0)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, F.coalesce(F.col(in_col), F.lit(value)))
        elif strategy == "forward_fill":
            from pyspark.sql.window import Window
            window = Window.orderBy(F.monotonically_increasing_id()).rowsBetween(Window.unboundedPreceding, 0)
            for in_col, out_col in zip(self.inputs, self.outputs):
                df = df.withColumn(out_col, F.last(F.col(in_col), ignorenulls=True).over(window))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return df