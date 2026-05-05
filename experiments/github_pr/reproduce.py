"""Reproduce Ray 2.55 IcebergDatasource pickle-safety bug.

Run with: python experiments/github_pr/reproduce.py

Expected: raises TypeError / PicklingError with "cannot pickle '_thread.lock'"
"""

import os, sys, tempfile, shutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import pickle
import pyarrow as pa
import ray
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, IntegerType, StringType, TimestamptzType
from pyiceberg.partitioning import PartitionSpec

# ── Create a tiny Iceberg table in /tmp ──────────────────────────────────────
TABLE_DIR = tempfile.mkdtemp(prefix="iceberg_repro_")
WAREHOUSE = "file://" + TABLE_DIR

catalog = load_catalog("default", type="sql",
                       uri=f"sqlite:///{TABLE_DIR}/catalog.db",
                       warehouse=WAREHOUSE)
try:
    catalog.create_namespace("default")
except Exception:
    pass

schema = Schema(
    NestedField(1, "id", IntegerType(), required=False),
    NestedField(2, "value", StringType(), required=False),
)

table = catalog.create_table_if_not_exists("default.repro", schema=schema)
table.append(pa.table({
    "id": pa.array([1, 2, 3], type=pa.int32()),
    "value": pa.array(["a", "b", "c"], type=pa.string()),
}))
print(f"Iceberg table created: default.repro  (3 rows)")

# ── Try to pickle a lazy read_iceberg() Dataset ──────────────────────────────
ray.init(ignore_reinit_error=True)

ds = ray.data.read_iceberg(
    table_identifier="default.repro",
    catalog_kwargs={"type": "sql",
                    "uri": f"sqlite:///{TABLE_DIR}/catalog.db",
                    "warehouse": WAREHOUSE},
)

# Force .table to be populated (triggers PyIceberg connection)
list(ds.iter_batches(batch_size=2))

print("Before fix — testing pickle...")
try:
    pickle.dumps(ds)
    print("SUCCESS: already pickle-safe")
except (TypeError, pickle.PicklingError, AttributeError) as e:
    print(f"BUG REPRODUCED: {e}")

# ── Apply fix ──
from ray.data._internal.datasource.iceberg_datasource import IcebergDatasource
def _pickle_safe_getstate(self):
    state = self.__dict__.copy()
    state["_table"] = None
    state["_plan_files"] = None
    return state
IcebergDatasource.__getstate__ = _pickle_safe_getstate

print("\nAfter fix — testing pickle...")
try:
    pickle.dumps(ds)
    print("FIX WORKED: dataset is now pickle-safe!")
except (TypeError, pickle.PicklingError, AttributeError) as e:
    print(f"STILL BROKEN: {e}")

ray.shutdown()
shutil.rmtree(TABLE_DIR, ignore_errors=True)
print("Done.")
