# xdas

xdas is a python library built around xarray that allows to work with DAS data 
(Distributed Acoustic Sensing).

## Installation

First install the dependencies:

    conda install dask h5py icoords netcdf4 numpy scipy xarray

Then install ```icoords``` wich is a pluging for xarray:

    git clone https://github.com/atrabattoni/icoords.git
    (cd icoords && pip install -e .)

Finally install ```xdas```:

    git clone https://github.com/atrabattoni/xdas.git
    (cd xdas && pip install -e .)

## Update

Pulling the change from github suffices:

    (cd xdas && git pull)
    (cd icoords && git pull)

## Usage

Here how to read a febus file with some decimation:

```python
from xdas.io.febus import read
fname = "path_to_febus_file.h5"
ixarr = read(fname, decimation=10)
```
The file will load coordinates as interpolated coordinates. To load then in memory and get a classic xarray DataArray object:

```python
xarr = ixarr.load_icoords()
```

To write and read the file using CF conventions:

```python
from icoords import InterpolatedDataArray
ixarr.to_netcdf("path.nc")
ixarr = InterpolatedDataArray.from_netcdf("path.nc")
```
