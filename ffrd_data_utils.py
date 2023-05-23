import dask.array as da
import numpy as np
import rioxarray
import xarray as xr
from scipy.stats import norm, genextreme
import zarr

from glob import glob
import os
from typing import Callable, List, Optional


SAMPLE_WIDTH = 1024
SAMPLE_HEIGHT = 1024
DISK_CHUNK_WIDTH = 512
DISK_CHUNK_HEIGHT = 512
CHUNK_WIDTH = 16
CHUNK_HEIGHT = 16
X1 = 2048
Y1 = 2048
X2 = X1 + SAMPLE_WIDTH
Y2 = Y1 + SAMPLE_HEIGHT
RESAMPLE_SIZE = 10000
BOOTSTRAP_SAMPLES = 10


def load_terrain(path: str, sample: bool = False) -> xr.DataArray:
    da = rioxarray.open_rasterio(path, chunks=(1, DISK_CHUNK_WIDTH, DISK_CHUNK_HEIGHT), masked=True).squeeze(drop=True)
    if sample:
        da = da[X1:X2, Y1:Y2]
    return da


def tifs_to_zarr(wsel_tifs_path: str, terrain_tif_path: str, zarr_out: str, sample: bool = False):
    wsel_tifs = glob(wsel_tifs_path)
    wsel_tifs.sort()

    store = zarr.DirectoryStore(zarr_out)

    terrain_da = load_terrain(terrain_tif_path, sample=sample)
    terrain_ds = terrain_da.to_dataset(name="terrain")
    terrain_ds.to_zarr(store=store, mode="w")

    for r, tif in enumerate(wsel_tifs):
        print(r, tif)
        wsel_da = rioxarray.open_rasterio(tif, masked=True).squeeze(drop=True)
        if sample:
            wsel_da = wsel_da[X1:X2, Y1:Y2]
        wsel_da = wsel_da.chunk({"x": DISK_CHUNK_WIDTH, "y": DISK_CHUNK_HEIGHT})
        wsel_da = wsel_da.expand_dims({"r": [r]})
        wsel_da.attrs = {}
        wsel_ds = wsel_da.to_dataset(name="wsel")
        append_dim = None if r == 0 else "r"
        wsel_ds.to_zarr(store=store,
                        mode="a",
                        consolidated=True,
                        append_dim=append_dim)


def rechunk_wsel_zarr(zarr_in: str, zarr_out: str):
    ds = xr.open_zarr(zarr_in)
    # https://stackoverflow.com/questions/67476513
    del ds["wsel"].encoding["chunks"]
    rechunked_ds = ds.chunk({"r": len(ds.r), "x": DISK_CHUNK_WIDTH, "y": DISK_CHUNK_HEIGHT})
    rechunked_ds.to_zarr(zarr_out, mode="w")


def depth(wsel: xr.DataArray, terrain: xr.DataArray) -> xr.DataArray:
    terrain = terrain.expand_dims({"r": wsel.r})
    depth_da = wsel - terrain
    depth_da = depth_da.fillna(0.0)
    return depth_da


def depth_da_to_zarr(depth_da: xr.DataArray, zarr_out: str,
                     spatial_ref: Optional[xr.DataArray] = None):
    depth_ds = depth_da.to_dataset(name="depth")
    depth_ds['spatial_ref'] = spatial_ref
    depth_ds.to_zarr(zarr_out, mode="w")


def norm_percentile(data: np.ndarray, p: float):
    mu, sigma = norm.fit(data)
    percentile = norm.ppf(p, loc=mu, scale=sigma)
    return percentile


def get_percentiles(data: np.ndarray, pcts: List[float], dist: Callable):
    params = dist.fit(data)
    return [dist.ppf(p, *params) for p in pcts]


def lower_conf_interval(data: np.ndarray, conf_interval: float = 90) -> float:
    return np.percentile(data, (100 - conf_interval) / 2)


def upper_conf_interval(data: np.ndarray, conf_interval: float = 90) -> float:
    return np.percentile(data, 100 - (100 - conf_interval) / 2)


def percentile_conf_interval(data: np.ndarray, pcts: List[float], dist: Callable,
                             conf_interval: float = 90, n_samples: int = BOOTSTRAP_SAMPLES,
                             weights: Optional[List[float]] = None) -> List[float]:
    resampled_data = np.random.choice(data, size=RESAMPLE_SIZE, replace=True, p=weights)
    percentiles = get_percentiles(resampled_data, pcts, dist)

    bootstrap_sample = np.random.choice(data, size=(data.shape[0], n_samples),
                                        p=weights, replace=True)
    bootstrap_percentiles = np.apply_along_axis(get_percentiles, 0, bootstrap_sample, pcts, dist)
    lower = np.apply_along_axis(lower_conf_interval, 1, bootstrap_percentiles, conf_interval)
    upper = np.apply_along_axis(upper_conf_interval, 1, bootstrap_percentiles, conf_interval)
    return [percentiles, lower, upper]


def depth_quantiles(depth: xr.DataArray, aeps: List[float],
                    spatial_ref: Optional[xr.DataArray] = None,
                    weights: Optional[List[float]] = None) -> xr.Dataset:
    probs = [1 - aep for aep in aeps]
    weights_adj = np.array(weights) / np.sum(weights)
    result = da.apply_along_axis(
        percentile_conf_interval,
        0,
        depth,
        probs,
        genextreme,
        shape=(3, len(probs)),
        dtype=np.float32,
        weights=weights_adj,
    )
    result_da = xr.DataArray(result, dims=["result", "y", "x", "aep"],
                                coords={
                                    "result": ["percentile", "lower_ci", "upper_ci"],
                                    "y": depth.y,
                                    "x": depth.x,
                                    "aep": aeps,
                                },
    )
    dataset = result_da.to_dataset(dim="result")
    if spatial_ref is not None:
        dataset["spatial_ref"] = spatial_ref
    return dataset


def norm_percentile_conf_interval(data: np.ndarray,
                                  pct: float, n_samples: int = BOOTSTRAP_SAMPLES,
                                  conf_interval: float = 90,
                                  weights: Optional[List[float]] = None) -> List[float]:
    resampled_data = np.random.choice(data, size=RESAMPLE_SIZE, replace=True, p=weights)
    percentile = norm_percentile(resampled_data, pct)

    bootstrap_sample = np.random.choice(data, size=(data.shape[0], n_samples),
                                        p=weights, replace=True)
    bootstrap_percentiles = np.apply_along_axis(norm_percentile, 0, bootstrap_sample, pct)
    lower = np.percentile(bootstrap_percentiles, (100 - conf_interval) / 2)
    upper = np.percentile(bootstrap_percentiles, 100 - (100 - conf_interval) / 2)
    return [percentile, lower, upper]


def depth_quantile(depth: xr.DataArray, aep: float,
                   spatial_ref: Optional[xr.DataArray] = None,
                   weights: Optional[List[float]] = None) -> xr.Dataset:
    p = 1 - aep
    weights_adj = np.array(weights) / np.sum(weights)
    result = da.apply_along_axis(
        norm_percentile_conf_interval,
        0,
        depth,
        p,
        shape=(3, ),
        dtype=np.float32,
        weights=weights_adj,
    )
    result_da = xr.DataArray(result, dims=["result", "y", "x"],
                             coords={
                                 "result": ["depth", "lower_ci", "upper_ci"],
                                 "y": depth.y,
                                 "x": depth.x,
                            },
    )
    dataset = result_da.to_dataset(dim="result")
    if spatial_ref is not None:
        dataset["spatial_ref"] = spatial_ref
    return dataset