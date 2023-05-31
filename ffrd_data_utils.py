import dask.array as da
import numpy as np
import rioxarray
import xarray as xr
from scipy.stats import norm, genextreme, rv_continuous
import zarr

from glob import glob
import os
from typing import Callable, List, Optional, Tuple


np.random.seed(42)

# Chunk size for working with initial raster data.
INITIAL_CHUNK_WIDTH = 512
INITIAL_CHUNK_HEIGHT = 512

# Chunk size for working with subsequent Zarr datasets.
# Smaller chunk sizes are better for parallel processing.
CHUNK_WIDTH = 16
CHUNK_HEIGHT = 16

# Size and location of the raster data to sample from.
# Helpful for dev/testing, since the full dataset is pretty large.
SAMPLE_HEIGHT = 1024
SAMPLE_WIDTH = 1024
X1 = 2048
Y1 = 2048
X2 = X1 + SAMPLE_WIDTH
Y2 = Y1 + SAMPLE_HEIGHT

# Number of samples to draw from the weighted raster data for bootsrapping
RESAMPLE_SIZE = 10000

# Number of bootstrap samples to create for estimating confidence intervals
BOOTSTRAP_SAMPLES = 10


def load_terrain(path: str, sample: bool = False) -> xr.DataArray:
    """
    Load the terrain data from a GeoTIFF file.

    Args:
        path: Path to the GeoTIFF file.
        sample: If True, only load a subset of the data for testing purposes.

    Returns:
        The terrain data as an xarray DataArray.
    """
    da = rioxarray.open_rasterio(path, chunks=(1, INITIAL_CHUNK_WIDTH, INITIAL_CHUNK_HEIGHT),
                                 masked=True).squeeze(drop=True)
    if sample:
        da = da[X1:X2, Y1:Y2]
    return da


def tifs_to_zarr(wsel_tifs_path: str, terrain_tif_path: str, zarr_out: str, sample: bool = False):
    """
    Convert the WSEL and terrain data from GeoTIFF files to a Zarr dataset.
    
    Args:
        wsel_tifs_path: Path to the WSEL GeoTIFF files.
        terrain_tif_path: Path to the terrain GeoTIFF file.
        zarr_out: Path to the output Zarr dataset.
        sample: If True, only load a subset of the data for testing purposes.
    """
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
        wsel_da = wsel_da.chunk({"x": INITIAL_CHUNK_WIDTH, "y": INITIAL_CHUNK_HEIGHT})
        wsel_da = wsel_da.expand_dims({"r": [r]})
        wsel_da.attrs = {}
        wsel_ds = wsel_da.to_dataset(name="wsel")
        append_dim = None if r == 0 else "r"
        wsel_ds.to_zarr(store=store,
                        mode="a",
                        consolidated=True,
                        append_dim=append_dim)


def rechunk_wsel_zarr(zarr_in: str, zarr_out: str):
    """
    Rechunk the WSEL Zarr dataset to a smaller chunk size.
    
    Args:
        zarr_in: Path to the input Zarr dataset.
        zarr_out: Path to the output Zarr dataset.
    """
    ds = xr.open_zarr(zarr_in)
    del ds["wsel"].encoding["chunks"]  # https://stackoverflow.com/questions/67476513
    rechunked_ds = ds.chunk({"r": len(ds.r), "x": INITIAL_CHUNK_WIDTH, "y": INITIAL_CHUNK_HEIGHT})
    rechunked_ds.to_zarr(zarr_out, mode="w")


def depth(wsel: xr.DataArray, terrain: xr.DataArray) -> xr.DataArray:
    """
    Calculate the depth of flooding from the WSEL and terrain data.
    Expands the terrain data to match the shape of the WSEL data, in 
    order to calculate the depth for each layer.
    
    Args:
        wsel: The WSEL data.
        terrain: The terrain data.
        
    Returns:
        The depth of flooding as an xarray DataArray.
    """
    terrain = terrain.expand_dims({"r": wsel.r})
    depth_da = wsel - terrain
    depth_da = depth_da.fillna(0.0)
    return depth_da


def depth_da_to_zarr(depth_da: xr.DataArray, zarr_out: str,
                     spatial_ref: Optional[xr.DataArray] = None):
    """
    Convert the depth of flooding data to a Zarr dataset.
    
    Args:
        depth_da: The depth of flooding data.
        zarr_out: Path to the output Zarr dataset.
        spatial_ref: The spatial reference data.
    """
    depth_ds = depth_da.to_dataset(name="depth")
    depth_ds['spatial_ref'] = spatial_ref
    depth_ds.to_zarr(zarr_out, mode="w")


# def norm_percentile(data: np.ndarray, p: float):
#     mu, sigma = norm.fit(data)
#     percentile = norm.ppf(p, loc=mu, scale=sigma)
#     return percentile


# def get_percentiles(data: np.ndarray, pcts: List[float], dist: rv_continuous):
#     params = dist.fit(data)
#     return [dist.ppf(p, *params) for p in pcts]


def lower_conf_interval(data: np.ndarray, conf_interval: float = 90) -> float:
    """
    Calculate the lower confidence interval for a given array of values.
    
    Args:
        data: The array of values.
        conf_interval: The confidence interval, in percent (0 to 100).
    
    Returns:
        The lower confidence interval value.
    """
    return np.percentile(data, (100 - conf_interval) / 2)


def upper_conf_interval(data: np.ndarray, conf_interval: float = 90) -> float:
    """
    Calculate the upper confidence interval for a given array of values.
    
    Args:
        data: The array of values.
        conf_interval: The confidence interval, in percent (0 to 100).
    
    Returns:
        The upper confidence interval value.
    """
    return np.percentile(data, 100 - (100 - conf_interval) / 2)


def zero_inflated_percentiles(data: np.ndarray, pcts: List[float],
                              dist: rv_continuous) -> List[float]:
    """
    Calculate the flood depth percentiles for a given list of AEP values. 

    Args:
        data: The array of flood depth values at a single location.
        pcts: The list of percentile values to estimate (0.0 to 1.0).
        dist: The distribution to use for estimating the percentiles.

    Returns:
        The estimated flood depths for the given percentiles.
    """
    # Calculate the portion of non-zero values
    prob_zero = np.sum(data == 0.0) / data.size

    if not any(pct > prob_zero for pct in pcts):
        # If all of the percentiles are below the portion of non-zero values,
        # then all results are 0.0
        return [0.0] * len(pcts)

    idx_nonzero = np.where(data > 0)
    data_nonzero = data[idx_nonzero]
    params = dist.fit(data_nonzero)

    results = []
    for pct in pcts:
        if pct > prob_zero:
            # Calculate the percentile of the non-zero values
            pct_nonzero = (pct - prob_zero) / (1.0 - prob_zero)
            result = dist.ppf(pct_nonzero, *params)
            result = 0.0 if result < 0.0 else result  # Ensure result is not negative
            results.append(result)
        else:
            results.append(0.0)
    return results


def percentile_conf_interval(data: np.ndarray, pcts: List[float], dist: rv_continuous,
                             conf_interval: float = 90, n_samples: int = BOOTSTRAP_SAMPLES,
                             weights: Optional[List[float]] = None) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate the flood depth percentiles for a given list of AEP values.
    Also calculate the confidence intervals for the percentiles using bootstrapping.
    
    Args:
        data: The array of flood depth values at a single location.
        pcts: The list of percentile values to estimate (0.0 to 1.0).
        dist: The distribution to use for estimating the percentiles.
        conf_interval: The confidence interval, in percent (0 to 100).
        n_samples: The number of bootstrap samples to create.
        weights: The weights to use for sampling the data.
        
    Returns:
        The estimated flood depths for the given percentiles, along with the
        lower and upper confidence intervals.
    """
    resampled_data = np.random.choice(data, size=RESAMPLE_SIZE, replace=True, p=weights)
    percentiles = zero_inflated_percentiles(resampled_data, pcts, dist)

    bootstrap_sample = np.random.choice(data, size=(RESAMPLE_SIZE, n_samples),
                                        p=weights, replace=True)
    bootstrap_percentiles = np.apply_along_axis(zero_inflated_percentiles, 0,
                                                bootstrap_sample, pcts, dist)
    lower = np.apply_along_axis(lower_conf_interval, 1, bootstrap_percentiles, conf_interval)
    upper = np.apply_along_axis(upper_conf_interval, 1, bootstrap_percentiles, conf_interval)
    return percentiles, lower, upper


def depth_quantiles(depth: xr.DataArray, aeps: List[float],
                    spatial_ref: Optional[xr.DataArray] = None,
                    weights: Optional[List[float]] = None) -> xr.Dataset:
    """
    Calculate the flood depth percentiles for a given list of AEP values.
    Also calculate the confidence intervals for the percentiles using bootstrapping.
    
    Args:
        depth: The depth of flooding data.
        aeps: The list of AEP values to estimate (0.0 to 1.0).
        spatial_ref: The spatial reference data.
        weights: The weights to use for sampling the data.
        
    Returns:
        The estimated flood depths for the given AEP values, along with the
        lower and upper confidence intervals, as an xarray Dataset.
    """
    probs = [1 - aep for aep in aeps]
    weights_adj = np.array(weights) / np.sum(weights)
    result = da.apply_along_axis(
        percentile_conf_interval,
        0,
        depth,
        probs,
        norm,
        shape=(3, len(probs)),
        dtype=np.float32,
        weights=weights_adj,
    )
    result_da = xr.DataArray(result, dims=["result", "aep", "y", "x"],
                                coords={
                                    "result": ["depth", "lower_ci", "upper_ci"],
                                    "y": depth.y,
                                    "x": depth.x,
                                    "aep": aeps,
                                },
    )
    dataset = result_da.to_dataset(dim="result")
    if spatial_ref is not None:
        dataset["spatial_ref"] = spatial_ref
    return dataset


# def norm_percentile_conf_interval(data: np.ndarray,
#                                   pct: float, n_samples: int = BOOTSTRAP_SAMPLES,
#                                   conf_interval: float = 90,
#                                   weights: Optional[List[float]] = None) -> List[float]:
#     resampled_data = np.random.choice(data, size=RESAMPLE_SIZE, replace=True, p=weights)
#     percentile = norm_percentile(resampled_data, pct)

#     bootstrap_sample = np.random.choice(data, size=(data.shape[0], n_samples),
#                                         p=weights, replace=True)
#     bootstrap_percentiles = np.apply_along_axis(norm_percentile, 0, bootstrap_sample, pct)
#     lower = np.percentile(bootstrap_percentiles, (100 - conf_interval) / 2)
#     upper = np.percentile(bootstrap_percentiles, 100 - (100 - conf_interval) / 2)
#     return [percentile, lower, upper]


# def depth_quantile(depth: xr.DataArray, aep: float,
#                    spatial_ref: Optional[xr.DataArray] = None,
#                    weights: Optional[List[float]] = None) -> xr.Dataset:
#     p = 1 - aep
#     weights_adj = np.array(weights) / np.sum(weights)
#     result = da.apply_along_axis(
#         norm_percentile_conf_interval,
#         0,
#         depth,
#         p,
#         shape=(3, ),
#         dtype=np.float32,
#         weights=weights_adj,
#     )
#     result_da = xr.DataArray(result, dims=["result", "y", "x"],
#                              coords={
#                                  "result": ["depth", "lower_ci", "upper_ci"],
#                                  "y": depth.y,
#                                  "x": depth.x,
#                             },
#     )
#     dataset = result_da.to_dataset(dim="result")
#     if spatial_ref is not None:
#         dataset["spatial_ref"] = spatial_ref
#     return dataset

