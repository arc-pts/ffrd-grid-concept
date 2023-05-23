import dask.array as da
import numpy as np
import rioxarray
import xarray as xr
from scipy.stats import norm
import zarr

from glob import glob
import os
from typing import List, Optional


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
    # da = rioxarray.open_rasterio(path, masked=True).squeeze(drop=True)
    if sample:
        da = da[X1:X2, Y1:Y2]
    return da


# def wsel_tifs_to_zarr2(path: str, zarr_out: str, sample: bool = False):
#     """
#     Load wsel tifs into a zarr dataset. Note that this is not chunked along the r dimension,
#     in order to allow for appending to the dataset.
#     """
#     wsel_tifs = glob(path)
#     wsel_tifs.sort()

#     store = zarr.DirectoryStore(zarr_out)
#     root = zarr.group(store, overwrite=True)

#     for r, tif in enumerate(wsel_tifs):
#         print(r, tif)
#         # wsel_da = rioxarray.open_rasterio(tif, chunks=(1, DISK_CHUNK_WIDTH, DISK_CHUNK_HEIGHT), masked=True).squeeze(drop=True)
#         wsel_da = rioxarray.open_rasterio(tif, masked=True).squeeze(drop=True)
#         wsel_da = wsel_da.chunk({"x": DISK_CHUNK_WIDTH, "y": DISK_CHUNK_HEIGHT})
#         wsel_da = wsel_da.expand_dims({"r": [r]})
#         if sample:
#             wsel_da = wsel_da[X1:X2, Y1:Y2]
#         wsel_da.attrs = {}
#         wsel_ds = wsel_da.to_dataset(name="wsel")
#         mode = "w" if r == 0 else "a"
#         append_dim = None if r == 0 else "r"
#         wsel_ds.to_zarr(store=store,
#                         mode=mode,
#                         consolidated=True,
#                         append_dim=append_dim)


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


# def wsel_tifs_to_zarr(path: str, zarr_out: str, sample: bool = False):
#     wsel_tifs = glob(path)
#     wsel_tifs.sort()

#     print("Opening tifs...")
#     wsel_layers = []
#     for tif in wsel_tifs:
#         wsel_da = rioxarray.open_rasterio(tif, chunks=(1, CHUNK_WIDTH, CHUNK_HEIGHT), masked=True).squeeze(drop=True)
#         print(wsel_da)
#         if sample:
#             wsel_da = wsel_da[X1:X2, Y1:Y2]
#         wsel_layers.append(wsel_da)

#     # combine layers along new dimension
#     print("Concatenating layers...")
#     wsel_da = xr.concat(wsel_layers, dim="r")

#     # chunk along new dimension
#     print("Chunking...")
#     wsel_da = wsel_da.chunk({"r": len(wsel_layers), "x": CHUNK_WIDTH, "y": CHUNK_HEIGHT})

#     # write to zarr
#     print("Writing to zarr...")
#     wsel_ds = wsel_da.to_dataset(name="wsel")
#     wsel_ds.to_zarr(zarr_out, mode="w")

def depth(wsel: xr.DataArray, terrain: xr.DataArray) -> xr.DataArray:
    # def depth_func(_wsel: np.ndarray, _terrain: np.ndarray):
    #     print(_wsel.shape, _terrain.shape)
    #     return _terrain - _wsel

    terrain = terrain.expand_dims({"r": wsel.r})

    # depth_da = xr.apply_ufunc(
    #     depth_func,
    #     wsel,
    #     terrain,
    #     input_core_dims=[["r"], ["r"]],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_dtypes=[np.float32],
    # )

    depth_da = wsel - terrain
    depth_da = depth_da.fillna(0.0)
    return depth_da


def depth_da_to_zarr(depth_da: xr.DataArray, zarr_out: str,
                     spatial_ref: Optional[xr.DataArray] = None):
    depth_ds = depth_da.to_dataset(name="depth")
    depth_ds['spatial_ref'] = spatial_ref
    depth_ds.to_zarr(zarr_out, mode="w")


def norm_percentile_conf_interval(data: np.ndarray,
                                  pct: float, n_samples: int = BOOTSTRAP_SAMPLES,
                                  conf_interval: float = 90,
                                  weights: Optional[List[float]] = None) -> List[float]:
    resampled_data = np.random.choice(data, size=RESAMPLE_SIZE, replace=True, p=weights)
    mu, sigma = norm.fit(resampled_data)
    percentile = norm.ppf(pct, loc=mu, scale=sigma)

    bootstrap_sample = np.random.choice(data, size=(data.shape[0], n_samples),
                                        p=weights, replace=True)
    bootstrap_percentiles = np.apply_along_axis(norm_percentile, 0, bootstrap_sample, pct)
    lower = np.percentile(bootstrap_percentiles, (100 - conf_interval) / 2)
    upper = np.percentile(bootstrap_percentiles, 100 - (100 - conf_interval) / 2)
    # bootstrap_percentiles = []
    # for _ in range(n_samples):
    #     bootstrap_sample = np.random.choice(data, size=data.shape, replace=True)

    #     mu_bootstrap, sigma_bootstrap = norm.fit(bootstrap_sample)

    #     bootstrap_percentile = norm.ppf(p, loc=mu_bootstrap, scale=sigma_bootstrap)
    #     bootstrap_percentiles.append(bootstrap_percentile)
    
    # bootstrap_percentiles = np.array(bootstrap_percentiles)
    # lower = np.percentile(bootstrap_percentiles, (100 - conf_interval) / 2)
    # upper = np.percentile(bootstrap_percentiles, 100 - (100 - conf_interval) / 2)
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


def norm_percentile(data: np.ndarray, p: float):
    mu, sigma = norm.fit(data)
    percentile = norm.ppf(p, loc=mu, scale=sigma)
    return percentile


def multi_norm_percentile_conf_interval(data: np.ndarray, pcts: List[float],
                                        n_samples: int = BOOTSTRAP_SAMPLES,
                                        conf_interval = 90):
    mu, sigma = norm.fit(data)
    result = []
    for pct in pcts:
        quantile = norm.ppf(pct, loc=mu, scale=sigma)
        bootstrap_sample = np.random.choice(data, size=(data.shape, n_samples), replace=True)
        bootstrap_percentiles = np.apply_along_axis(norm_percentile, 0, bootstrap_sample, pct)
        lower = np.percentile(bootstrap_percentiles, (100 - conf_interval) / 2)
        upper = np.percentile(bootstrap_percentiles, 100 - (100 - conf_interval) / 2)
        result.append([quantile, lower, upper])
    # for pct in pcts:
    #     percentile = norm.ppf(pct, loc=mu, scale=sigma)
    #     bootstrap_percentiles = []
    #     for _ in range(n_samples):
    #         bootstrap_sample = np.random.choice(data, size=data.shape, replace=True)

    #         mu_bootstrap, sigma_bootstrap = norm.fit(bootstrap_sample)

    #         bootstrap_percentile = norm.ppf(pct, loc=mu_bootstrap, scale=sigma_bootstrap)
    #         bootstrap_percentiles.append(bootstrap_percentile)
        
    #     bootstrap_percentiles = np.array(bootstrap_percentiles)
    #     lower = np.percentile(bootstrap_percentiles, (100 - conf_interval) / 2)
    #     upper = np.percentile(bootstrap_percentiles, 100 - (100 - conf_interval) / 2)
    #     result.append([percentile, lower, upper])
    return result


def depth_multi_quantile(depth: xr.DataArray, aeps: List[float],
                         spatial_ref: Optional[xr.DataArray] = None) -> xr.Dataset:
    pcts = [1 - aep for aep in aeps]
    result = da.apply_along_axis(
        multi_norm_percentile_conf_interval,
        0,
        depth,
        pcts,
        shape=(len(aeps), 3, ),
        dtype=np.float32,
    )
    result_da = xr.DataArray(result, dims=["aep", "result", "y", "x"],
                             coords={
                                 "aep": aeps,
                                 "result": ["depth", "lower_ci", "upper_ci"],
                                 "y": depth.y,
                                 "x": depth.x,
                            },
    )
    dataset = result_da.to_dataset(dim="result")
    if spatial_ref is not None:
        dataset["spatial_ref"] = depth.spatial_ref
    return dataset