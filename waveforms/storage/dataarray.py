from __future__ import annotations

import pickle
import base64
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
            _dataarray_to_hdf(self, file, key)

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
            if _group_is_dataarray(file[key]):
                return _dataarray_from_hdf_group(file[key])
            else:
                raise ValueError(
                    f"Cannot convert hdf group '{key}' to xarray.")

    @staticmethod
    def is_xarray(path_or_buf: Union[str, Path, BytesIO, BufferedReader],
                  key: str) -> bool:
        """Check if the group is an xarray group.

        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.

        Returns:
            bool: True if the group is an xarray group.
        """
        with h5py.File(path_or_buf, 'r') as file:
            return _group_is_xarray(file[key])


def _coords_to_hdf(data: dict, group: h5py.Group) -> None:
    """Write the coords to an HDF group.

    Parameters:
        data (dict): The dictionary of coordinates to write.
        group (h5py.Group): The HDF group to write to.
    """
    if 'attrs' in data:
        for k, v in data['attrs'].items():
            group.attrs[k] = v
    for name, coord in data.items():
        d = group.create_dataset(name, data=coord['data'])
        if 'attrs' in coord:
            for n, a in coord['attrs'].items():
                d.attrs[n] = a


def _coords_from_hdf_group(group: h5py.Group) -> dict:
    """Read the coords from an HDF group.

    Parameters:
        group (h5py.Group): The HDF group to read from.

    Returns:
        dict: The dictionary of coordinates.
    """
    ret = {}
    for name, dset in group.items():
        ret[name] = {'data': dset[...], 'attrs': {}, 'dims': (name, )}
        for n, a in dset.attrs.items():
            ret[name]['attrs'][n] = a
    return ret


def _attrs_to_hdf(attrs: dict, group_or_dataset: Union[h5py.Group,
                                                       h5py.Dataset]) -> None:
    """Write the attributes to an HDF group or dataset.

    Parameters:
        attrs (dict): The dictionary of attributes to write.
        group_or_dataset (h5py.Group or h5py.Dataset): The HDF group or dataset
            to write to.
    """
    for k, v in attrs.items():
        try:
            group_or_dataset.attrs[k] = v
        except TypeError:
            group_or_dataset.attrs[k] = "pickle://base64:" + base64.b64encode(
                pickle.dumps(v)).decode()


def _attrs_from_hdf_group_or_dataset(
        group_or_dataset: Union[h5py.Group, h5py.Dataset]) -> dict:
    """Read the attributes from an HDF group or dataset.

    Parameters:
        group_or_dataset (h5py.Group or h5py.Dataset): The HDF group or dataset
            to read from.

    Returns:
        dict: The dictionary of attributes.
    """
    ret = {}
    for k, v in group_or_dataset.attrs.items():
        if isinstance(v, str) and v.startswith('pickle://base64:'):
            ret[k] = pickle.loads(base64.b64decode(
                v[len('pickle://base64:'):]))
        else:
            ret[k] = v
    return ret


def _dataarray_to_hdf(data: DataArray, file: h5py.File, key: str) -> None:
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
    _attrs_to_hdf({'dims': dct['dims'], 'name': dct['name']}, g)
    d = g.create_dataset('data', data=np.asarray(dct['data']))
    _attrs_to_hdf(dct['attrs'], d)
    _coords_to_hdf(dct['coords'], g.create_group('coords'))


def _dataarray_from_hdf_group(group: h5py.Group) -> DataArray:
    """Retrieve DataArray object stored in HDF group.

    Parameters:
        group (h5py.Group): The HDF group to read from.

    Returns:
        DataArray: The DataArray object.
    """
    coords = _coords_from_hdf_group(group['coords'])
    attrs = _attrs_from_hdf_group_or_dataset(group['data'])
    data = np.asarray(group['data'][...])
    dct = {'coords': coords, 'attrs': attrs, 'data': data}
    dct.update(_attrs_from_hdf_group_or_dataset(group))
    return DataArray.from_dict(dct)


def _group_is_dataarray(group: h5py.Group) -> bool:
    """Check if the group contains an xarray dataset.
    
    Parameters:
        group (h5py.Group): The HDF group to read from.

    Returns:
        bool: True if the group contains an xarray dataset.
    """
    return ('dims' in group.attrs and 'name' in group.attrs and 'data' in group
            and 'coords' in group)


class Dataset(xr.Dataset):
    def to_hdf(self, path_or_buf: Union[str, Path, BytesIO, BufferedWriter],
               key: str) -> None:
        """Write the contained data to an HDF file.
        
        In order to add another DataFrame or Series to an existing HDF file please
        use a different key.

        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.
        """
        with h5py.File(path_or_buf, 'a') as file:
            _dataset_to_hdf(self, file, key)

    @staticmethod
    def from_hdf(path_or_buf: Union[str, Path, BytesIO, BufferedReader],
                 key: str) -> Dataset:
        """Read from the store, close it if we opened it.

        Retrieve Dataset object stored in file.

        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.

        Returns:
            DataArray: The Dataset object.
        """
        with h5py.File(path_or_buf, 'r') as file:
            if _group_is_dataset(file[key]):
                return _dataset_from_hdf_group(file[key])
            else:
                raise ValueError(
                    f"Cannot convert hdf group '{key}' to xarray.")

    @staticmethod
    def is_xarray(path_or_buf: Union[str, Path, BytesIO, BufferedReader],
                  key: str) -> bool:
        """Check if the group is an xarray group.

        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.

        Returns:
            bool: True if the group is an xarray group.
        """
        with h5py.File(path_or_buf, 'r') as file:
            return _group_is_dataset(file[key])


def _dataset_to_hdf(data: Dataset, file: h5py.File, key: str) -> None:
    """Write the Dataset to an HDF file.

    Parameters:
        data (Dataset): The Dataset object to write.
        file (h5py.File): The HDF file to write to.
        key (str): Identifier for the group in the store.
    """
    if isinstance(data, dict):
        dct = data
    elif isinstance(data, xr.Dataset):
        dct = data.to_dict()
    else:
        raise TypeError(f'unsupported type {type(data)}, '
                        'only dict or DataArray allowed.')

    g = file.create_group(key)
    coords = g.create_group('coords')
    data_vars = g.create_group('data_vars')
    _attrs_to_hdf({'dims': dct['dims']}, coords)
    _attrs_to_hdf(dct['attrs'], g)
    _coords_to_hdf(dct['coords'], coords)
    data_dims = {}
    for k, v in dct['data_vars'].items():
        dataset = data_vars.create_dataset(k, data=np.asarray(v['data']))
        _attrs_to_hdf(v['attrs'], dataset)
        data_dims[k] = v['dims']
    _attrs_to_hdf(data_dims, data_vars)


def _dataset_from_hdf_group(group: h5py.Group) -> Dataset:
    """Retrieve Dataset object stored in HDF group.

    Parameters:
        group (h5py.Group): The HDF group to read from.

    Returns:
        Dataset: The Dataset object.
    """
    coords = _coords_from_hdf_group(group['coords'])
    attrs = _attrs_from_hdf_group_or_dataset(group)
    dims = _attrs_from_hdf_group_or_dataset(group['coords'])['dims']
    dct = {'coords': coords, 'dims': dims, 'attrs': attrs, 'data_vars': {}}
    data_dims = _attrs_from_hdf_group_or_dataset(group['data_vars'])
    for k, v in group['data_vars'].items():
        attrs = _attrs_from_hdf_group_or_dataset(v)
        data = np.asarray(v[...])
        dct['data_vars'][k] = {
            'dims': data_dims[k],
            'attrs': attrs,
            'data': data
        }
    return Dataset.from_dict(dct)


def _group_is_dataset(group: h5py.Group) -> bool:
    """Check if the group contains an xarray dataset.
    
    Parameters:
        group (h5py.Group): The HDF group to read from.

    Returns:
        bool: True if the group contains an xarray dataset.
    """
    return True or ('data_vars' in group and 'coords' in group
                    and 'dims' in group['coords'])
