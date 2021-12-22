import dask.array as da
import h5py
import numpy as np
import scipy.signal as sp
import xarray as xr
from icoords.core import InterpolatedCoordinates, InterpolatedDataArray
from icoords.interpolate import LinearCoordinate


def read(fname, decimation=None):
    """Read a febus A1-R file into a InterpolatedDataArray"""
    data, time, delta, name = read_hdf5(fname)
    data, time = trim_overlaps(data, time, delta)
    time_icoord = get_time_icoord(data, time, delta)
    offset_icoord = get_offset_icoord(data, delta)
    if decimation is not None:
        data, delta = decimate(data, delta, decimation)
    data = da.reshape(data, (-1, data.shape[2]))
    ixarr = pack(data, time_icoord, offset_icoord, name)
    return ixarr


def read_hdf5(fname):
    file = h5py.File(fname, "r")
    device_name, = list(file.keys())
    device = file[device_name]
    source = device["Source1"]
    time = np.asarray(source["time"])
    zone = source["Zone1"]
    name, = list(zone.keys())
    delta = {
        "time": zone.attrs["Spacing"][1] / 1000.0,
        "offset": zone.attrs["Spacing"][0],
    }
    data = da.from_array(zone[name])
    name = to_snakecase(name)
    return data, time, delta, name


def trim_overlaps(data, time, delta):
    noverlap = data.shape[1] // 4
    data = data[:, noverlap:-noverlap, :]
    time = time + noverlap * delta["time"]
    return data, time


def get_time_icoord(data, time, delta):
    starts = np.rint(1e6 * time).astype("datetime64[us]")
    size = data.shape[1]
    step = np.rint(1e6 * delta["time"]).astype("timedelta64[us]")
    return to_linear_coordinate(starts, size, step)


def get_offset_icoord(data, delta):
    starts = 0.0
    size = data.shape[2]
    step = delta["offset"]
    return to_linear_coordinate(starts, size, step)


def decimate(data, delta, factor):
    if data.shape[1] % factor != 0:
        raise ValueError(
            "the length of the blocks must be a multitple of the decimation "
            "factor")
    if 20 * factor > data.shape[1] // 2:
        raise ValueError(
            "the length of the FIR coeffs must be at most half the length of "
            "the blocks")
    data = da.blockwise(
        lambda x: sp.decimate(x, factor, ftype="fir", axis=1),
        "ijk",
        data,
        "ijk",
        adjust_chunks={"j": lambda n: n // factor},
        dtype=data.dtype,
    )
    delta["time"] = factor * delta["time"]
    return data, delta


def pack(data, time_icoord, offset_icoord, name):
    dims = ("time", "offset")
    icoords = InterpolatedCoordinates({
        "time": time_icoord,
        "offset": offset_icoord})
    icoords["time"].simplify(np.timedelta64(0, "us"))
    data_array = xr.DataArray(data, dims=dims, name=name)
    ixarr = InterpolatedDataArray(data_array, icoords)
    return ixarr


def to_snakecase(name):
    name = "".join(["_"+c.lower() if c.isupper()
                   else c for c in name]).lstrip("_")
    return name


def to_linear_coordinate(starts, sizes, steps):
    starts, sizes, steps = np.broadcast_arrays(starts, sizes, steps)
    start_values = starts
    end_values = starts + steps * (sizes - 1)
    start_indices = np.insert(np.cumsum(sizes)[:-1], 0, 0)
    end_indices = np.cumsum(sizes) - 1
    tie_values = np.stack((start_values, end_values)).T.reshape(-1)
    tie_indices = np.stack((start_indices, end_indices)).T.reshape(-1)
    time_coordinate = LinearCoordinate(tie_indices, tie_values)
    return time_coordinate
