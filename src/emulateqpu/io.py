"""Canonical identities and atomic, integrity-checked application records."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def read_json(path: str | os.PathLike[str]) -> Any:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out: raise ValueError('Duplicate JSON key: '+k)
            out[k] = v
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def atomic_json(path: str | os.PathLike[str], value: Any, *, sort_keys: bool = True) -> None:
    """Write JSON to a fresh temporary sibling, then replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=False)+'\n'
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def inspect_result(path):
    path = Path(path); path = path/'result.json' if path.is_dir() else path
    r = read_json(path)
    if r.get('schema') != 'emulateqpu.result.v1': raise ValueError('Unsupported result schema')
    for name, sha in r.get('payload_sha256', {}).items():
        p = path.parent/name
        if p.resolve().parent != path.parent.resolve() or p.is_symlink(): raise ValueError('Unsafe result member')
        if file_hash(p) != sha: raise ValueError('Result integrity mismatch: '+name)
    return r
