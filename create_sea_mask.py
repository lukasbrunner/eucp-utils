#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2018 Lukas Brunner (ETH Zurich)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Time-stamp: <2018-11-01 11:47:19 lukas>

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Creates a land-sea mask file based on a given input grid based on the
regionmask package.

Properties
----------
- Masks according to the grid cell center! I.e., grid cells with more than 50%
  land might be masked if their center is over the ocean and vice versa.
- The Caspian Sea is not masked
- Based on Natural Earth 1:110m

References
----------
https://www.naturalearthdata.com/
https://regionmask.readthedocs.io/en/stable/defined_landmask.html

"""
import os
import logging
import argparse
import regionmask
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# DEBUG: matplotlib bug; issue 1120
# https://github.com/SciTools/cartopy/issues/1120
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


def get_longitude_name(ds):
    """Get the name of the longitude dimension by CF unit"""
    lonn = []
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = ds.to_dataset()
    for dimn in ds.dims.keys():
        if ('units' in ds[dimn].attrs and
            ds[dimn].attrs['units'] in ['degree_east', 'degrees_east']):
            lonn.append(dimn)
    if len(lonn) == 1:
        return lonn[0]
    elif len(lonn) > 1:
        errmsg = 'More than one longitude coordinate found by unit.'
    else:
        errmsg = 'Longitude could not be identified by unit.'
    raise ValueError(errmsg)


def get_latitude_name(ds):
    """Get the name of the latitude dimension by CF unit"""
    latn = []
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = ds.to_dataset()
    for dimn in ds.dims.keys():
        if ('units' in ds[dimn].attrs and
            ds[dimn].attrs['units'] in ['degree_north', 'degrees_north']):
            latn.append(dimn)
    if len(latn) == 1:
        return latn[0]
    elif len(latn) > 1:
        errmsg = 'More than one latitude coordinate found by unit.'
    else:
        errmsg = 'Latitude could not be identified by unit.'
    raise ValueError(errmsg)


def get_variable_name(ds):
    """Try to get the main variable from a dataset.

    This function tries to get the main variable from a dataset by checking the
    following properties:
    - the variable must NOT be a coordinate variable
    - the variable name must NOT contain '_bounds' or '_bnds'
    - There MUST be exactly one variable fulfilling the above

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with only one main variable, e.g., tas (time, lat, lon)
    """

    varns = [varn for varn in set(ds.variables).difference(ds.coords)
             if '_bounds' not in varn and '_bnds' not in varn]
    if len(varns) != 1:
        errmsg = 'Unable to select a single variable from {}'.format(
            ', '.join(varns))
        raise ValueError(errmsg)
    return varns[0]


def flip_antimeridian(ds, to='Pacific', lonn=None):
    """
    Flip the antimeridian (i.e. longitude discontinuity) between Europe
    (i.e., [0, 360)) and the Pacific (i.e., [-180, 180)).

    Parameters:
    - ds (xarray.Dataset or .DataArray): Has to contain a single longitude
      dimension.
    - to='Pacific' (str, optional): Flip antimeridian to one of
      * 'Europe': Longitude will be in [0, 360)
      * 'Pacific': Longitude will be in [-180, 180)
    - lonn=None (str, optional): Name of the longitude dimension. If None it
      will be inferred by the CF convention standard longitude unit.

    Returns:
    same type as input ds
    """
    if lonn is None:
        lonn = get_longitude_name(ds)
    elif lonn not in ds.dims:
        errmsg = '{} not found in Dataset!'.format(lonn)
        raise ValueError(errmsg)

    if to.lower() == 'europe' and ds[lonn].min() >= 0:
        return ds  # already correct, do nothing
    elif to.lower() == 'pacific' and ds[lonn].max() < 180:
        return ds  # already correct, do nothing
    elif to.lower() == 'europe':
        ds = ds.assign_coords(**{lonn: (ds.lon % 360)})
    elif to.lower() == 'pacific':
        ds = ds.assign_coords(**{lonn: (((ds.lon + 180) % 360) - 180)})
    else:
        errmsg = 'to has to be one of [Europe | Pacific] not {}'.format(to)
        raise ValueError(errmsg)

    was_da = isinstance(ds, xr.core.dataarray.DataArray)
    if was_da:
        da_varn = ds.name
        ds = ds.to_dataset()

    idx = np.argmin(ds[lonn].data)
    varns = [varn for varn in ds.variables if lonn in ds[varn].dims]
    for varn in varns:
        if xr.__version__ > '0.10.8':
            ds[varn] = ds[varn].roll(**{lonn: -idx}, roll_coords=False)
        else:
            ds[varn] = ds[varn].roll(**{lonn: -idx})

    if was_da:
        return ds[da_varn]
    return ds


def get_mask(da):
    latn = get_latitude_name(da)
    lonn = get_longitude_name(da)
    da = flip_antimeridian(da, lonn=lonn)
    mask = regionmask.defined_regions.natural_earth.land_110.mask(da) == 0
    mask.name = 'land_sea_mask'
    mask.attrs = {
        'long_name': 'Land-sea mask',
        'description': ' '.join([
        'Based on Natural Earth: https://www.naturalearthdata.com;',
            'The Caspian Sea is not masked!']),
        'flag_values': '0, 1',
        'flag_meaning': 'ocean, land'}

    ds = mask.to_dataset()
    ds[latn].attrs = {'units': 'degree_north'}
    ds[lonn].attrs = {'units': 'degree_east'}
    return ds


def get_grid(fn, varn=None):
    """Read in the data"""
    da = xr.open_dataset(fn)
    if varn is None:
        varn = get_variable_name(da)
    return da[varn]


def plot(ds):
    """Optional function to show the mask"""
    ds = flip_antimeridian(ds)
    proj = ccrs.PlateCarree(central_longitude=0)
    fig, ax = plt.subplots(subplot_kw={'projection': proj})
    ds['land_sea_mask'].plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False)
    ax.coastlines()
    xx, yy = np.meshgrid(ds['lon'], ds['lat'])
    ax.scatter(xx, yy, s=1, color='k')
    plt.show()


def read_config():
    """Read user input"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='filename', type=str,
        help='A file with a latitude-longitude grid as basis for the mask')
    parser.add_argument(
        '--variable-name', '-v', dest='varn', default=None, type=str,
        help=' '.join([
            'If filename contains more than one variable varn has to be given',
            'and point to a variable depending on latitude-longitude.']))
    parser.add_argument(
        '--save-path' '-s', dest='save_path', default=None, type=str,
        help='Location to save the mask. Defaults to ./mask.nc')
    parser.add_argument(
        '--plot', '-p', dest='plot', action='store_true',
        help='Show a map for checking the mask.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig()
    args = read_config()
    da = get_grid(args.filename, varn=args.varn)
    ds = get_mask(da)

    if args.plot:
        plot(ds)

    if args.save_path is None:
        args.save_path = './land_sea_mask.nc'

    ds.to_netcdf(args.save_path)
