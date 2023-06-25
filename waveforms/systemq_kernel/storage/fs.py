import gzip
import hashlib
import pickle
import zlib
from pathlib import Path

import dill

DATAPATH = Path.home() / 'data'
CHUNKSIZE = 1024 * 1024 * 4  # 4 MB


def set_data_path(base_path: str) -> None:
    global DATAPATH
    DATAPATH = Path(base_path)


def get_data_path() -> Path:
    return DATAPATH


def _save_object(data) -> tuple[str, str]:
    try:
        data = pickle.dumps(data)
    except:
        data = dill.dumps(data)
    hashstr = hashlib.sha1(data).hexdigest()
    file = get_data_path(
    ) / 'objects' / hashstr[:2] / hashstr[2:4] / hashstr[4:]
    file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(file, 'wb') as f:
        f.write(data)
    return str('/'.join(file.parts[-4:])), hashstr


def _load_object(file: str) -> bytes:
    with gzip.open(get_data_path() / file, 'rb') as f:
        try:
            data = pickle.load(f)
        except:
            f.seek(0)
            data = dill.load(f)
    return data


def _delete_object(file: str):
    file = get_data_path() / file
    file.unlink()


def save_chunk(data: bytes, compressed: bool = False) -> tuple[str, str]:
    if compressed:
        data = zlib.compress(data)
    hashstr = hashlib.sha1(data).hexdigest()
    file = get_data_path(
    ) / 'objects' / hashstr[:2] / hashstr[2:4] / hashstr[4:]
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'wb') as f:
        f.write(data)
    return str('/'.join(file.parts[-4:])), hashstr


def load_chunk(file: str, compressed: bool = False) -> bytes:
    with open(get_data_path() / file, 'rb') as f:
        data = f.read()
    if compressed:
        data = zlib.decompress(data)
    return data


def delete_chunk(file: str):
    file = get_data_path() / file
    file.unlink()
