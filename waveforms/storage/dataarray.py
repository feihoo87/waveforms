from __future__ import annotations

from io import BufferedReader, BufferedWriter, BytesIO
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import xarray as xr


class DataArray(xr.DataArray):
    def to_hdf(self, path_or_buf: Union[str, Path, BytesIO, BufferedWriter],
               key: str) -> None:
        """Write the contained data to an HDF5 file.
        
        In order to add another DataFrame or Series to an existing HDF file please
        use a different key.

        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.
        """
        with h5py.File(path_or_buf, 'a') as file:
            _to_hdf(self, file, key)

    @staticmethod
    def from_hdf(path_or_buf: Union[str, Path, BytesIO, BufferedReader],
                 key: str) -> DataArray:
        """Read from the store, close it if we opened it.

        Retrieve DataArray object stored in file.

        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.

        Returns:
            DataArray: The DataArray object.
        """
        with h5py.File(path_or_buf, 'r') as file:
            if _group_is_xarray(file[key]):
                return _from_hdf_group(file[key])
            else:
                raise ValueError(f"Cannot convert hdf group '{key}' to xarray.")


def _to_hdf(data: DataArray, file: h5py.File, key: str) -> None:
    """Write the DataArray to an HDF file.

    Parameters:
        data (DataArray): The DataArray object to write.
        file (h5py.File): The HDF file to write to.
        key (str): Identifier for the group in the store.
    """
    if isinstance(data, dict):
        dct = data
    elif isinstance(data, xr.DataArray):
        dct = data.to_dict()
    else:
        raise TypeError(f'unsupported type {type(data)}, '
                        'only dict or DataArray allowed.')

    g = file.create_group(key)
    d = g.create_dataset('data', data=np.asarray(dct['data']))

    g.attrs['dims'] = dct['dims']
    g.attrs['name'] = dct['name']

    if 'attrs' in dct:
        for k, v in dct['attrs'].items():
            d.attrs[k] = v

    if 'coords' in dct:
        for k, v in dct['coords'].items():
            d = g.create_dataset('/'.join(['coords', k]),
                                 data=np.asarray(v['data']))
            if 'attrs' in v:
                for n, a in v['attrs'].items():
                    d.attrs[n] = a


def _from_hdf_group(group: h5py.Group) -> DataArray:
    """Retrieve DataArray object stored in HDF group.

    Parameters:
        group (h5py.Group): The HDF group to read from.

    Returns:
        DataArray: The DataArray object.
    """
    coords = {name: group['coords'][name][...] for name in group['coords']}
    ret = DataArray(group['data'][...],
                    dims=group.attrs['dims'],
                    coords=coords,
                    name=group.attrs['name'])
    for name in group['data'].attrs:
        ret.attrs[name] = group['data'].attrs[name]
    for name in coords:
        for a in group['coords'][name].attrs:
            ret.coords[name].attrs[a] = group['coords'][name].attrs[a]
    return ret


def _group_is_xarray(group: h5py.Group) -> bool:
    """Check if the group contains an xarray dataset.
    
    Parameters:
        group (h5py.Group): The HDF group to read from.

    Returns:
        bool: True if the group contains an xarray dataset.
    """
    return ('dims' in group.attrs and 'name' in group.attrs and 'data' in group
            and 'coords' in group)
