import h5py
import geopandas as gpd
import numpy as np
import pandas as pd

from enum import Enum
from typing import List


def read_ras_2d_flow_area_names(hdf_file: str) -> List[str]:
    """
    Read the 2D flow area names from a HEC-RAS HDF file.

    Args:
        hdf_file (str): The path to the HEC-RAS HDF file.

    Returns:
        List[str]: The 2D flow area names.
    """
    names = []
    with h5py.File(hdf_file, 'r') as f:
        for row in f['Geometry']['2D Flow Areas']['Attributes']:
            names.append(row[0].decode('utf-8'))
    return names


def read_hecras_2d_points(hdf_file: str) -> np.ndarray:
    """
    Read the 2D points from a HEC-RAS HDF file.

    Args:
        hdf_file (str): The path to the HEC-RAS HDF file.
        flow_area_name (str): The name of the flow area to read.

    Returns:
        np.ndarray: The 2D points.
    """
    flow_area_names = read_ras_2d_flow_area_names(hdf_file)
    points = []
    with h5py.File(hdf_file, 'r') as f:
        for flow_area_name in flow_area_names:
            flow_area_points = f['Geometry']['2D Flow Areas'][flow_area_name]['Cells Center Coordinate'][:]
            points.append(flow_area_points)
    return np.concatenate(points, axis=0)


class RasSummaryOutputVar(Enum):
    CUMULATIVE_CELL_ITER = 'Cumulative Cell Iter'
    MAXIMUM_FACE_COURANT = 'Maximum Face Courant'
    MAXIMUM_FACE_SHEAR_STRESS = 'Maximum Face Shear Stress'
    MAXIMUM_FACE_VELOCITY = 'Maximum Face Velocity'
    MAXIMUM_WATER_SURFACE = 'Maximum Water Surface'
    MINIMUM_FACE_VELOCITY = 'Minimum Face Velocity'
    MINIMUM_WATER_SURFACE = 'Minimum Water Surface'


def read_hecras_2d_summary_output(hdf_file: str, output_var: RasSummaryOutputVar) -> np.ndarray:
    flow_area_names = read_ras_2d_flow_area_names(hdf_file)
    output = []
    with h5py.File(hdf_file, 'r') as f:
        for flow_area_name in flow_area_names:
            flow_area_output = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Summary Output']['2D Flow Areas'][flow_area_name][output_var.value][0]
            output.append(flow_area_output)
    return np.concatenate(output, axis=0)


def ras_hdf_2dpoints_to_geodataframe(hdf_file: str) -> gpd.GeoDataFrame:
    """
    Convert a 2D points (cell centers) of HEC-RAS HDF results file to a GeoDataFrame.

    Args:
        hdf_file (str): The path to the HEC-RAS HDF file.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame.
    """
    points = read_hecras_2d_points(hdf_file)
    wsel_max = read_hecras_2d_summary_output(hdf_file, RasSummaryOutputVar.MAXIMUM_WATER_SURFACE)
    wsel_min = read_hecras_2d_summary_output(hdf_file, RasSummaryOutputVar.MINIMUM_WATER_SURFACE)
    df = pd.DataFrame(points, columns=['x', 'y'])
    df['wsel_max'] = wsel_max
    df['wsel_min'] = wsel_min
    return df
    # gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    # return gdf.drop(columns=['x', 'y'])
