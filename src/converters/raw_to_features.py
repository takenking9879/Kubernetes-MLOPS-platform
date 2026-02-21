"""
Pure-Python equivalent of kafka_to_schema_features for online/real-time use.

Mirrors the Spark converter in spark_kafka_helper.py without any Spark dependency.
Used in Ray Serve to preprocess raw Kafka-style events before the NumPy DSL executor.

Spark reference (spark_kafka_helper.py):
    df.select(
        col("event_id"),
        col("properties.src_port").cast("long"),
        col("properties.dst_port").cast("long"),
        col("properties.protocol"),
        col("properties.packet_count").cast("long"),
        col("properties.conn_state"),
        col("properties.bytes_transferred").cast("double"),
        to_timestamp(col("timestamp")).cast("long"),   # <- epoch seconds
    )

Input (raw Kafka event dict):
    {
        "timestamp":  "2026-01-01T00:30:03",
        "event_id":   "some-uuid",
        "properties": {
            "src_port":          12345,
            "dst_port":          80,
            "protocol":          "TCP",
            "packet_count":      10,
            "conn_state":        "SF",
            "bytes_transferred": 1024.0,
        }
    }

Output (feature dict matching schema_features):
    {
        "event_id":          "some-uuid",
        "src_port":          12345,       # int (Long)
        "dst_port":          80,          # int (Long)
        "protocol":          "TCP",       # str
        "packet_count":      10,          # int (Long)
        "conn_state":        "SF",        # str
        "bytes_transferred": 1024.0,      # float (Double)
        "timestamp":         1735691403,  # int  (epoch seconds, matches Spark's cast("long"))
    }
"""

from datetime import datetime, timezone
from typing import Any, Dict


def raw_event_to_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw Kafka-style event dict to a feature dict.

    Mirrors Spark's kafka_to_schema_features():
      - Flattens properties.*
      - Converts timestamp string → epoch seconds (int), matching
        Spark's to_timestamp().cast("long")
      - Casts numeric fields to their correct types

    The resulting dict is ready to pass directly to NumpyPipelineExecutor.transform().
    """
    props = raw.get("properties") or {}
    ts    = raw.get("timestamp", "")

    return {
        "event_id":          str(raw.get("event_id", "")),
        "src_port":          int(props.get("src_port", 0)),
        "dst_port":          int(props.get("dst_port", 0)),
        "protocol":          str(props.get("protocol", "")),
        "packet_count":      int(props.get("packet_count", 0)),
        "conn_state":        str(props.get("conn_state", "")),
        "bytes_transferred": float(props.get("bytes_transferred", 0.0)),
        "timestamp":         _to_epoch_seconds(ts),
    }


def _to_epoch_seconds(ts: Any) -> int:
    """
    Convert various timestamp representations to UTC epoch seconds (int).

    Matches Spark's to_timestamp(col).cast("long") behaviour.
    """
    if isinstance(ts, (int, float)):
        return int(ts)

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp())

    s = str(ts).strip()
    if not s:
        return 0

    # Normalise 'Z' suffix → '+00:00'
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # Try common formats (naive first → treat as UTC, matching Spark default)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue

    # Fallback: handles ISO strings with explicit offset
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())
