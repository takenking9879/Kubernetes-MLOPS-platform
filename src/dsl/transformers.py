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


class CastTransformer(Transformer):
    """Cast a column to a different data type."""
    
    def get_type(self) -> str:
        return "cast_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        target_type = self.params.get("target_type", "string")
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        type_mapping = {
            "timestamp": TimestampType(),
            "double": DoubleType(),
            "integer": IntegerType(),
            "string": StringType(),
            "LongType": LongType()
        }
        
        spark_type = type_mapping.get(target_type, StringType())
        return df.withColumn(output_col, F.col(input_col).cast(spark_type))


class TemporalExtractor(Transformer):
    """Extract temporal components from timestamp columns."""
    
    def get_type(self) -> str:
        return "temporal_extractor"
    
    def transform(self, df: DataFrame) -> DataFrame:
        component = self.params.get("component", "hour")
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        if component == "hour":
            return df.withColumn(output_col, F.hour(input_col))
        elif component == "minute":
            return df.withColumn(output_col, F.minute(input_col))
        elif component == "second":
            return df.withColumn(output_col, F.second(input_col))
        elif component == "dayofweek":
            return df.withColumn(output_col, F.dayofweek(input_col))
        elif component == "dayofmonth":
            return df.withColumn(output_col, F.dayofmonth(input_col))
        elif component == "month":
            return df.withColumn(output_col, F.month(input_col))
        elif component == "year":
            return df.withColumn(output_col, F.year(input_col))
        elif component == "quarter":
            return df.withColumn(output_col, F.quarter(input_col))
        elif component == "weekofyear":
            return df.withColumn(output_col, F.weekofyear(input_col))
        else:
            raise ValueError(f"Unknown temporal component: {component}")


class ArithmeticTransformer(Transformer):
    """Apply arithmetic operations to columns."""
    
    def get_type(self) -> str:
        return "arithmetic_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        operation = self.params.get("operation", "add")
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        if operation == "add":
            value = self.params.get("value", 0)
            return df.withColumn(output_col, F.col(input_col) + F.lit(value))
        elif operation == "subtract":
            value = self.params.get("value", 0)
            return df.withColumn(output_col, F.col(input_col) - F.lit(value))
        elif operation == "multiply":
            value = self.params.get("value", 1)
            return df.withColumn(output_col, F.col(input_col) * F.lit(value))
        elif operation == "divide":
            value = self.params.get("value", 1)
            return df.withColumn(output_col, F.col(input_col) / F.lit(value))
        elif operation == "power":
            value = self.params.get("value", 2)
            return df.withColumn(output_col, F.pow(F.col(input_col), F.lit(value)))
        elif operation == "absolute":
            return df.withColumn(output_col, F.abs(F.col(input_col)))
        elif operation == "negate":
            return df.withColumn(output_col, -F.col(input_col))
        else:
            raise ValueError(f"Unknown operation: {operation}")


class ConditionalTransformer(Transformer):
    """Apply conditional logic to create new columns."""
    
    def get_type(self) -> str:
        return "conditional_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        condition = self.params.get("condition", "isin")
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        if condition == "isin":
            values = self.params.get("values", [])
            true_value = self.params.get("true_value", 1)
            false_value = self.params.get("false_value", 0)
            return df.withColumn(
                output_col,
                F.when(F.col(input_col).isin(values), true_value).otherwise(false_value)
            )
        elif condition == "greater_than":
            threshold = self.params.get("threshold", 0)
            true_value = self.params.get("true_value", 1)
            false_value = self.params.get("false_value", 0)
            return df.withColumn(
                output_col,
                F.when(F.col(input_col) > threshold, true_value).otherwise(false_value)
            )
        elif condition == "less_than":
            threshold = self.params.get("threshold", 0)
            true_value = self.params.get("true_value", 1)
            false_value = self.params.get("false_value", 0)
            return df.withColumn(
                output_col,
                F.when(F.col(input_col) < threshold, true_value).otherwise(false_value)
            )
        elif condition == "equals":
            value = self.params.get("value")
            true_value = self.params.get("true_value", 1)
            false_value = self.params.get("false_value", 0)
            return df.withColumn(
                output_col,
                F.when(F.col(input_col) == value, true_value).otherwise(false_value)
            )
        elif condition == "is_null":
            true_value = self.params.get("true_value", 1)
            false_value = self.params.get("false_value", 0)
            return df.withColumn(
                output_col,
                F.when(F.col(input_col).isNull(), true_value).otherwise(false_value)
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")


class CyclicTransformer(Transformer):
    """Apply cyclic encoding (sin/cos) for periodic features."""
    
    def get_type(self) -> str:
        return "cyclic_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        function = self.params.get("function", "sin")
        period = self.params.get("period", 24.0)
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        # Calculate angle: 2π * value / period
        angle = 2 * math.pi * F.col(input_col) / period
        
        if function == "sin":
            return df.withColumn(output_col, F.sin(angle))
        elif function == "cos":
            return df.withColumn(output_col, F.cos(angle))
        else:
            raise ValueError(f"Unknown function: {function}. Use 'sin' or 'cos'.")


class LogTransformer(Transformer):
    """Apply logarithmic transformations."""
    
    def get_type(self) -> str:
        return "log_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        log_type = self.params.get("log_type", "log1p")
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        if log_type == "log1p":
            return df.withColumn(output_col, F.log1p(F.col(input_col)))
        elif log_type == "log":
            return df.withColumn(output_col, F.log(F.col(input_col)))
        elif log_type == "log10":
            return df.withColumn(output_col, F.log10(F.col(input_col)))
        elif log_type == "log2":
            return df.withColumn(output_col, F.log2(F.col(input_col)))
        else:
            raise ValueError(f"Unknown log_type: {log_type}")


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
        bins = self.params.get("bins", [])
        labels = self.params.get("labels", None)
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        if not bins:
            raise ValueError("Bins parameter is required")
        
        # Use Spark's bucketizer-like logic with when/otherwise
        expr = None
        for i in range(len(bins) - 1):
            condition = (F.col(input_col) >= bins[i]) & (F.col(input_col) < bins[i + 1])
            label = labels[i] if labels and i < len(labels) else i
            
            if expr is None:
                expr = F.when(condition, label)
            else:
                expr = expr.when(condition, label)
        
        # Handle values outside bins
        default_label = labels[-1] if labels and len(labels) > len(bins) - 1 else len(bins) - 1
        expr = expr.otherwise(default_label)
        
        return df.withColumn(output_col, expr)


class ClipTransformer(Transformer):
    """Clip values to a specified range."""
    
    def get_type(self) -> str:
        return "clip_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        min_value = self.params.get("min_value")
        max_value = self.params.get("max_value")
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        expr = F.col(input_col)
        
        if min_value is not None:
            expr = F.when(expr < min_value, min_value).otherwise(expr)
        
        if max_value is not None:
            expr = F.when(expr > max_value, max_value).otherwise(expr)
        
        return df.withColumn(output_col, expr)


class FillNATransformer(Transformer):
    """Fill missing values with a constant or strategy."""
    
    def get_type(self) -> str:
        return "fillna_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        strategy = self.params.get("strategy", "constant")
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        if strategy == "constant":
            value = self.params.get("value", 0)
            return df.withColumn(
                output_col,
                F.coalesce(F.col(input_col), F.lit(value))
            )
        elif strategy == "forward_fill":
            # Forward fill using window function
            from pyspark.sql.window import Window
            window = Window.orderBy(F.monotonically_increasing_id()).rowsBetween(Window.unboundedPreceding, 0)
            return df.withColumn(
                output_col,
                F.last(F.col(input_col), ignorenulls=True).over(window)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")