"""Strict, auditable external FRF ingestion; does not fit or evaluate a model.

The exported NPZ remains compatible with the original SparseLDV-QA CLI.
No complex FRF is reconstructed from magnitude alone: real/imaginary parts
and physically identified axes are required.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd


class InputContractError(ValueError):
    """Raised when an external dataset violates the declared physical contract."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _schema(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise InputContractError('schema must be a JSON object')
    return cfg


def _require(cfg, key, values):
    v = cfg.get(key)
    if v not in values:
        raise InputContractError(f'{key} must be explicitly declared as one of {sorted(values)}')
    return v


def _nonempty_id(s, field):
    if s.isna().any() or s.astype(str).str.strip().eq('').any():
        raise InputContractError(f'missing {field} identifiers')
    return s.astype(str).str.strip()


def _finish(input_path, schema_path, output_path, arrays, cfg, details):
    out = Path(output_path)
    if out.resolve() == Path(input_path).resolve():
        raise InputContractError('cannot overwrite raw source data')
    out.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed with an open file keeps caller's exact output name.
    with out.open('wb') as f:
        np.savez_compressed(f, **arrays)
    report = {
        'schema_version': 'sparseldv-user-data-v1',
        'source_file': Path(input_path).name,
        'source_sha256': _sha256(input_path),
        'mapping_file': Path(schema_path).name,
        'mapping_sha256': _sha256(schema_path),
        'output_file': out.name,
        'output_sha256': _sha256(out),
        'coordinate_unit_output': 'm',
        'frequency_unit_output': 'Hz',
        'complex_response': cfg.get('complex_format', 'real_imag') + '_to_rectangular',
        'algorithm_execution': False,
        'input_groups_are_independent': 'not established by import',
        **details,
    }
    manifest = out.with_suffix(out.suffix + '.manifest.json')
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def convert_csv(input_path, schema_path, output_path):
    """Convert long-format complex FRF table to training/truth-compatible NPZ.

    Each sample contains the same unique point x frequency grid. A single
    explicitly declared holdout sample is not used for sensor selection.
    """
    cfg = _schema(schema_path)
    units = _require(cfg, 'coordinate_unit', {'m', 'mm'})
    funits = _require(cfg, 'frequency_unit', {'Hz', 'kHz'})
    response_unit = cfg.get('response_unit')
    if not isinstance(response_unit, str) or not response_unit.strip():
        raise InputContractError('response_unit is required; do not guess a physical FRF unit')
    truth_id = str(cfg.get('truth_sample', '')).strip()
    if not truth_id:
        raise InputContractError('truth_sample must identify an explicit held-out sample')
    mapping = cfg.get('columns', {})
    fmt = cfg.get('complex_format', 'real_imag')
    if fmt not in {'real_imag', 'magnitude_phase'}:
        raise InputContractError('complex_format must be real_imag or magnitude_phase')
    fields = ('sample', 'point', 'frequency', 'x', 'y') + (('real', 'imag') if fmt == 'real_imag' else ('magnitude', 'phase'))
    mapping = {k: mapping.get(k, k) for k in fields + ('z', 'coherence')}
    df = pd.read_csv(input_path)
    missing = [mapping[k] for k in fields if mapping[k] not in df.columns]
    if missing:
        raise InputContractError(f'missing required columns: {missing}')
    has_z = mapping['z'] in df.columns
    has_coh = mapping['coherence'] in df.columns
    if has_coh and cfg.get('quality_kind') != 'magnitude_squared_coherence':
        raise InputContractError('coherence field requires quality_kind=magnitude_squared_coherence')
    df = df.copy()
    df['__sid'] = _nonempty_id(df[mapping['sample']], 'sample')
    df['__pid'] = _nonempty_id(df[mapping['point']], 'point')
    measure_cols = [mapping[x] for x in fields if x not in ('sample', 'point')]
    if has_z: measure_cols.append(mapping['z'])
    if has_coh: measure_cols.append(mapping['coherence'])
    for name in measure_cols:
        df[name] = pd.to_numeric(df[name], errors='coerce')
    if not np.isfinite(df[measure_cols].to_numpy(float)).all():
        raise InputContractError('non-numeric, NaN or infinite FRF/frequency/coordinate/coherence')
    if fmt == 'magnitude_phase':
        if cfg.get('phase_unit') not in {'degree', 'radian'} or cfg.get('phase_sign') not in (1, -1) or not str(cfg.get('phase_reference', '')).strip():
            raise InputContractError('polar input requires phase_unit, phase_sign (+1/-1), and phase_reference')
        mag = df[mapping['magnitude']].to_numpy(float)
        if np.any(mag < 0):
            raise InputContractError('linear magnitude must be nonnegative (dB magnitude is not accepted)')
        phase = df[mapping['phase']].to_numpy(float) * cfg['phase_sign']
        if cfg['phase_unit'] == 'degree': phase = np.deg2rad(phase)
        df['__converted_real'] = mag * np.cos(phase)
        df['__converted_imag'] = mag * np.sin(phase)
        mapping['real'], mapping['imag'] = '__converted_real', '__converted_imag'
    df['__frequency'] = df[mapping['frequency']].to_numpy(float) * (1000 if funits == 'kHz' else 1)
    if (df['__frequency'] <= 0).any():
        raise InputContractError('frequencies must be positive')
    if df.duplicated(['__sid', '__pid', '__frequency']).any():
        raise InputContractError('duplicate sample/point/frequency rows; average repeats explicitly upstream')
    coords = ['x', 'y'] + (['z'] if has_z else [])
    factor = 0.001 if units == 'mm' else 1.0
    for c in coords:
        df['__' + c] = df[mapping[c]].to_numpy(float) * factor
    ids = sorted(df['__pid'].unique().tolist())
    frequencies = np.sort(df['__frequency'].unique())
    sample_ids = sorted(df['__sid'].unique().tolist())
    if len(ids) < 3 or len(frequencies) < 2 or len(sample_ids) < 2:
        raise InputContractError('need >=3 points, >=2 frequencies, and >=2 independent sample IDs')
    if truth_id not in sample_ids:
        raise InputContractError('declared truth_sample is absent')
    train_ids = [x for x in sample_ids if x != truth_id]
    if 'training_samples' in cfg:
        train_ids = [str(x) for x in cfg['training_samples']]
        if len(set(train_ids)) != len(train_ids) or not train_ids or truth_id in train_ids or set(train_ids) != set(sample_ids) - {truth_id}:
            raise InputContractError('training_samples must contain every non-holdout sample once')
    # The same point must describe the same coordinates in every record.
    point_xyz = []
    for pid in ids:
        xyz = df.loc[df['__pid'] == pid, ['__' + k for k in coords]].to_numpy(float)
        if not np.allclose(xyz, xyz[0], rtol=0, atol=1e-8):
            raise InputContractError(f'contradictory coordinates for physical point {pid}')
        point_xyz.append(xyz[0])
    coords_array = np.asarray(point_xyz, float)
    if len(np.unique(coords_array, axis=0)) != len(ids):
        raise InputContractError('distinct point IDs share coordinates; resolve physical identity')
    qvalues = []
    cubes = []
    for sid in train_ids + [truth_id]:
        part = df.loc[df['__sid'] == sid]
        if len(part) != len(ids) * len(frequencies):
            raise InputContractError(f'sample {sid} has a missing or extra point/frequency grid cell')
        pivot = part.set_index(['__pid', '__frequency']).reindex(pd.MultiIndex.from_product([ids, frequencies]))
        if pivot[mapping['real']].isna().any():
            raise InputContractError(f'sample {sid} does not have the common complete grid')
        arr = pivot[mapping['real']].to_numpy(float) + 1j * pivot[mapping['imag']].to_numpy(float)
        cubes.append(arr.reshape(len(ids), len(frequencies)))
        if has_coh:
            qvalues.append(pivot[mapping['coherence']].to_numpy(float).reshape(len(ids), -1))
    arr = np.asarray(cubes, complex)
    if np.any(np.linalg.norm(arr, axis=(1, 2)) == 0):
        raise InputContractError('zero-energy scan')
    arrays = {'training': arr[:-1], 'truth': arr[-1], 'coordinates': coords_array,
              'point_ids': np.asarray(ids, dtype=str), 'frequency_hz': frequencies,
              'training_sample_ids': np.asarray(train_ids, dtype=str), 'truth_sample_id': np.asarray(truth_id)}
    if has_coh:
        qq = np.asarray(qvalues)
        if np.any((qq < 0) | (qq > 1)):
            raise InputContractError('coherence values must be within [0, 1]')
        arrays['quality'] = np.median(qq[:-1], axis=(0, 2))  # training only: no holdout leakage
    groups = cfg.get('sample_groups')
    if groups is not None:
        if not isinstance(groups, dict) or set(groups) != set(sample_ids) or not all(str(x).strip() for x in groups.values()):
            raise InputContractError('sample_groups must map all samples to nonempty group IDs')
        if cfg.get('enforce_group_holdout', True) and str(groups[truth_id]) in {str(groups[t]) for t in train_ids}:
            raise InputContractError('train/holdout groups overlap; grouping leakage detected')
        arrays['training_group_ids'] = np.asarray([str(groups[x]) for x in train_ids])
        arrays['truth_group_id'] = np.asarray(str(groups[truth_id]))
    return _finish(input_path, schema_path, output_path, arrays, cfg, {
        'input_type': 'csv_long', 'n_training_scans': len(train_ids), 'n_points': len(ids),
        'n_frequencies': len(frequencies), 'truth_sample': truth_id,
        'quality_provided': has_coh, 'quality_computed_from_training_only': has_coh, 'response_unit': response_unit,
        'group_holdout_checked': groups is not None and cfg.get('enforce_group_holdout', True),
    })


def convert_npz(input_path, schema_path, output_path):
    """Validate existing NumPy tensors; insist on independently defined axes."""
    cfg = _schema(schema_path)
    unit = _require(cfg, 'coordinate_unit', {'m', 'mm'})
    funit = _require(cfg, 'frequency_unit', {'Hz', 'kHz'})
    if not str(cfg.get('response_unit', '')).strip():
        raise InputContractError('response_unit must be declared')
    with np.load(input_path, allow_pickle=False) as z:
        required = {'training', 'truth', 'coordinates', 'frequency_hz', 'point_ids', 'training_sample_ids', 'truth_sample_id'}
        if not required <= set(z.files):
            raise InputContractError(f'NPZ missing {sorted(required - set(z.files))}')
        arrays = {k: np.array(z[k]) for k in required}
        for k in ('quality', 'training_group_ids', 'truth_group_id'):
            if k in z:
                arrays[k] = np.array(z[k])
    if not cfg.get('allow_real_only', False) and (not np.iscomplexobj(arrays['training']) or not np.iscomplexobj(arrays['truth'])):
        raise InputContractError('NPZ must contain complex arrays; declare allow_real_only=true for zero-imaginary data')
    a = np.asarray(arrays['training'], complex)
    b = np.asarray(arrays['truth'], complex)
    xyz = np.asarray(arrays['coordinates'], float)
    ids = np.asarray(arrays['point_ids']).astype(str)
    hz = np.asarray(arrays['frequency_hz'], float)
    if a.ndim != 3 or a.shape[0] < 1 or b.shape != a.shape[1:] or xyz.ndim != 2 or xyz.shape[0] != a.shape[1] or xyz.shape[1] not in (2, 3) or a.shape[1] < 3 or a.shape[2] < 2 or hz.shape != (a.shape[2],) or ids.shape != (a.shape[1],):
        raise InputContractError('NPZ axis mismatch: training(sample,point,freq), truth(point,freq), coordinates(point,2/3)')
    sample_ids = np.asarray(arrays['training_sample_ids']).astype(str)
    heldout_id = str(np.asarray(arrays['truth_sample_id']).item())
    if sample_ids.shape != (len(a),) or len(set(sample_ids)) != len(sample_ids) or heldout_id in set(sample_ids) or not heldout_id.strip() or any(not x.strip() for x in sample_ids):
        raise InputContractError('explicit unique training IDs and a different holdout ID required')
    if ('training_group_ids' in arrays) != ('truth_group_id' in arrays):
        raise InputContractError('both train and holdout group IDs required when grouping is provided')
    if 'training_group_ids' in arrays:
        train_groups = np.asarray(arrays['training_group_ids']).astype(str)
        truth_group = str(np.asarray(arrays['truth_group_id']).item())
        if train_groups.shape != (len(a),) or not truth_group or not all(train_groups) or (cfg.get('enforce_group_holdout', True) and truth_group in set(train_groups)):
            raise InputContractError('group identifiers invalid or train/holdout leakage')
    if len(set(ids)) != len(ids) or any(not x.strip() for x in ids) or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(xyz).all() or not np.isfinite(hz).all() or np.any(np.diff(hz) <= 0) or hz[0] <= 0:
        raise InputContractError('invalid point identity, nonfinite arrays or nonincreasing frequencies')
    xyz *= 0.001 if unit == 'mm' else 1
    if len(np.unique(xyz, axis=0)) != len(xyz):
        raise InputContractError('duplicate coordinates')
    arrays['training'] = a
    arrays['truth'] = b
    arrays['coordinates'] = xyz
    arrays['point_ids'] = ids
    arrays['frequency_hz'] = hz * (1000 if funit == 'kHz' else 1)
    if 'quality' in arrays:
        q = np.asarray(arrays['quality'], float)
        if q.shape != (len(ids),) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)) or cfg.get('quality_kind') != 'magnitude_squared_coherence':
            raise InputContractError('quality must be (point,) in [0,1] with declared coherence meaning')
        arrays['quality'] = q
    return _finish(input_path, schema_path, output_path, arrays, cfg, {
        'input_type': 'npz_tensor', 'n_training_scans': len(a), 'n_points': len(ids),
        'n_frequencies': len(hz), 'response_unit': cfg['response_unit'],
        'quality_provided': 'quality' in arrays,
        'group_holdout_checked': 'training_group_ids' in arrays and cfg.get('enforce_group_holdout', True),
    })


def convert(input_path, schema_path, output_path):
    suffix = Path(input_path).suffix.lower()
    if suffix == '.csv': return convert_csv(input_path, schema_path, output_path)
    if suffix == '.npz': return convert_npz(input_path, schema_path, output_path)
    raise InputContractError('supported input: .csv or .npz')
