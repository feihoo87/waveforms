from __future__ import annotations

import base64
import pickle
from io import BufferedReader, BufferedWriter, BytesIO
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import xarray as xr


def _pyobj_to_hdf(obj: object, key: str,
                  group_or_file: Union[h5py.Group, h5py.File]) -> None:
    """Write a Python object to an HDF group.
    Parameters:
        obj (object): The Python object to write.
        key (str): Identifier for the group in the store.
        group_or_file (h5py.Group or h5py.File): The HDF group or file to write
            to.
    """
    if isinstance(obj, np.ndarray):
        data = obj
        dtype = 'ndarray'
    elif isinstance(obj, bytes):
        data = np.frombuffer(obj, dtype=np.uint8)
        dtype = 'bytes'
    elif isinstance(obj, str):
        data = np.frombuffer(obj.encode(), dtype=np.uint8)
        dtype = 'string'
    else:
        data = np.frombuffer(pickle.dumps(obj), dtype=np.uint8)
        dtype = 'pickle'

    ds = group_or_file.create_dataset(key, data=data)
    ds.attrs['_TYPE_'] = dtype


def _pyobj_from_hdf(ds: h5py.Dataset) -> object:
    """Read a Python object from an HDF group.
    Parameters:
        ds (h5py.Dataset): The HDF group to read from.
    Returns:
        object: The Python object.
    """
    if '_TYPE_' not in ds.attrs:
        dtype = 'ndarray'
    else:
        dtype = ds.attrs['_TYPE_']

    if dtype == 'ndarray':
        return ds[...]
    elif dtype == 'bytes':
        return ds[...].tobytes()
    elif dtype == 'string':
        return ds[...].tobytes().decode()
    elif dtype == 'pickle':
        return pickle.loads(ds[...].tobytes())
    else:
        raise TypeError(f'unsupported type {dtype}, '
                        f'expected bytes, string, ndarray or pickle')


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


def _dataarray_to_hdf(data: xr.DataArray, file: h5py.File, key: str) -> None:
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
    _pyobj_to_hdf(dct['dims'], 'dims', g)
    _pyobj_to_hdf(dct['name'], 'name', g)
    d = g.create_dataset('data', data=np.asarray(dct['data']))
    _attrs_to_hdf(dct['attrs'], d)
    _coords_to_hdf(dct['coords'], g.create_group('coords'))
    g.attrs['_TYPE_'] = 'DataArray'


def _dataarray_from_hdf_group(group: h5py.Group) -> xr.DataArray:
    """Retrieve DataArray object stored in HDF group.
    Parameters:
        group (h5py.Group): The HDF group to read from.
    Returns:
        DataArray: The DataArray object.
    """
    coords = _coords_from_hdf_group(group['coords'])
    attrs = _attrs_from_hdf_group_or_dataset(group['data'])
    data = np.asarray(group['data'][...])
    dct = {
        'coords': coords,
        'attrs': attrs,
        'data': data,
        'dims': _pyobj_from_hdf(group['dims']),
        'name': _pyobj_from_hdf(group['name'])
    }
    return xr.DataArray.from_dict(dct)


def _group_is_dataarray(group: h5py.Group) -> bool:
    """Check if the group contains an xarray dataset.
    
    Parameters:
        group (h5py.Group): The HDF group to read from.
    Returns:
        bool: True if the group contains an xarray dataset.
    """
    return '_TYPE_' in group.attrs and group.attrs['_TYPE_'] == 'DataArray'


def _dataset_to_hdf(data: xr.Dataset, file: h5py.File, key: str) -> None:
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
    _pyobj_to_hdf(dct['dims'], 'dims', g)
    _pyobj_to_hdf(dct['attrs'], 'attrs', g)
    _coords_to_hdf(dct['coords'], coords)
    data_dims = {}
    for k, v in dct['data_vars'].items():
        dataset = data_vars.create_dataset(k, data=np.asarray(v['data']))
        _attrs_to_hdf(v['attrs'], dataset)
        data_dims[k] = v['dims']
    _attrs_to_hdf(data_dims, data_vars)
    g.attrs['_TYPE_'] = 'Dataset'


def _dataset_from_hdf_group(group: h5py.Group) -> xr.Dataset:
    """Retrieve Dataset object stored in HDF group.
    Parameters:
        group (h5py.Group): The HDF group to read from.
    Returns:
        Dataset: The Dataset object.
    """
    coords = _coords_from_hdf_group(group['coords'])
    attrs = _pyobj_from_hdf(group['attrs'])
    dims = _pyobj_from_hdf(group['dims'])
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
    return xr.Dataset.from_dict(dct)


def _group_is_dataset(group: h5py.Group) -> bool:
    """Check if the group contains an xarray dataset.
    
    Parameters:
        group (h5py.Group): The HDF group to read from.
    Returns:
        bool: True if the group contains an xarray dataset.
    """
    return '_TYPE_' in group.attrs and group.attrs['_TYPE_'] == 'Dataset'


def dataarray_to_hdf(data_array: xr.DataArray,
                     path_or_buf: Union[str, Path, BytesIO,
                                        BufferedWriter], key: str) -> None:
    """Write the contained data to an HDF5 file.
        
        In order to add another DataFrame or Series to an existing HDF file please
        use a different key.
        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.
        """
    with h5py.File(path_or_buf, 'a') as file:
        _dataarray_to_hdf(data_array, file, key)


def dataarray_from_hdf(path_or_buf: Union[str, Path, BytesIO, BufferedReader],
                       key: str) -> xr.DataArray:
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
            raise ValueError(f"Cannot convert hdf group '{key}' to xarray.")


def is_dataarray(path_or_buf: Union[str, Path, BytesIO, BufferedReader],
                 key: str) -> bool:
    """Check if the group is an xarray group.
        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.
        Returns:
            bool: True if the group is an xarray group.
        """
    with h5py.File(path_or_buf, 'r') as file:
        return _group_is_dataarray(file[key])


def dataset_to_hdf(dataset: xr.Dataset, path_or_buf: Union[str, Path, BytesIO,
                                                           BufferedWriter],
                   key: str) -> None:
    """Write the contained data to an HDF file.
        
        In order to add another DataFrame or Series to an existing HDF file please
        use a different key.
        Parameters:
            path_or_buf: Path to the HDF file or an open file-like object.
            key (str): Identifier for the group in the store.
        """
    with h5py.File(path_or_buf, 'a') as file:
        _dataset_to_hdf(dataset, file, key)


def dataset_from_hdf(path_or_buf: Union[str, Path, BytesIO, BufferedReader],
                     key: str) -> xr.Dataset:
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
            raise ValueError(f"Cannot convert hdf group '{key}' to xarray.")


def is_dataset(path_or_buf: Union[str, Path, BytesIO, BufferedReader],
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


__all__ = [
    'dataarray_from_hdf', 'dataarray_to_hdf', 'dataset_from_hdf',
    'dataset_to_hdf', 'is_dataarray', 'is_dataset'
]
