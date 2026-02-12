import ray

def maybe_sample_train_ds(ds: ray.data.Dataset, sample_fraction: float | None, seed: int) -> ray.data.Dataset:
    if sample_fraction is None:
        return ds
    frac = float(sample_fraction)
    if frac >= 1.0 or frac <= 0.0:
        return ds
    if hasattr(ds, "random_sample"):
        # do sampling; repartition afterwards to control parallelism
        ds = ds.random_sample(frac, seed=seed)
        return ds
    return ds