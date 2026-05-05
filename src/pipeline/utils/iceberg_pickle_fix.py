"""Fix for Ray 2.55 Iceberg datasource pickle-safety.

Ray 2.55's ``IcebergDatasource`` caches a live PyIceberg ``Table`` object
(``self._table``) whose ``.io`` property (S3FileIO) carries ``_thread.lock``
objects from internal thread pools.  When Ray Train tries to serialize the
datasource to ship it to workers, pickle raises::

    cannot pickle '_thread.lock' object

The fix: add ``__getstate__`` that drops ``_table`` and ``_plan_files`` before
serialization.  Workers lazily recreate them from ``_catalog_kwargs`` (a plain
dict — fully serializable).

Apply once during pipeline init, before any ``read_iceberg()`` call.
"""

import logging

logger = logging.getLogger(__name__)


def apply_iceberg_pickle_fix():
    """Monkey-patch Ray's IcebergDatasource to be pickle-safe.

    Idempotent — safe to call multiple times.
    """
    try:
        from ray.data._internal.datasource.iceberg_datasource import IcebergDatasource
    except ImportError:
        logger.debug("IcebergDatasource not found — pickle fix skipped.")
        return

    if hasattr(IcebergDatasource, "_iceberg_pickle_fix_applied"):
        return

    def _pickle_safe_getstate(self):
        state = self.__dict__.copy()
        state["_table"] = None
        state["_plan_files"] = None
        return state

    IcebergDatasource.__getstate__ = _pickle_safe_getstate
    IcebergDatasource._iceberg_pickle_fix_applied = True
    logger.info("IcebergDatasource pickle fix applied — lazy datasets are now serializable.")
