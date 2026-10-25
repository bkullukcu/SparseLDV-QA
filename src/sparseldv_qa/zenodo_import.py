"""Zenodo v5 Polytec H1 amplitude/phase ZIP -> complex group-labelled NPZ.

Designed for DOI 10.5281/zenodo.22770267. No experimental data are
bundled. Strictly fail on ambiguous labels, unmatched points or frequencies.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

__all__ = ["scan_archive", "convert_archive", "read_polytec", "load_coordinates"]


def _kind(path):
    parts = PurePosixPath(path).parts
    exact = [p.lower() for p in parts[:-1]]
    if "amplitude" in exact:
        return "amplitude"
    if "phase" in exact:
        return "phase"
    filename = PurePosixPath(path).stem.lower()
    if re.search(r"(?<![a-z])amplitude(?![a-z])", filename):
        return "amplitude"
    if re.search(r"(?<![a-z])phase(?![a-z])", filename):
        return "phase"
    return None


def _key(path):
    parts = list(PurePosixPath(path).parts)
    parts = [p for p in parts if p.lower() not in ("amplitude", "phase")]
    stem = parts[-1]
    stem = re.sub(r"(?i)amplitude|phase", "", stem)
    stem = re.sub(r"[_\- ]{2,}", "_", stem)
    parts[-1] = stem
    return "/".join(parts).lower()


def _parse_point(filename):
    name = PurePosixPath(filename).stem
    m = re.search(r"(?i)(?:scan\s*point|point|pt|pnt)[_\-\s#]*(\d+)(?!\d)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_\- ])P[_\- ]?(\d+)(?:[_\- ]|$)", name, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:[_\- ])(\d+)$", name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot infer physical point index from filename {filename!r}; provide a manual manifest.csv")


def _parse_state(path):
    parts = [p for p in PurePosixPath(path).parts[:-1] if p.lower() not in ("amplitude", "phase", "00_raw_exports")]
    if not parts:
        raise ValueError(f"Cannot infer torque-state folder from {path!r}; provide a manual manifest.csv")
    return parts[-1]


def scan_archive(zip_path, manifest_path):
    """Create a REVIEWABLE mapping; no measurements are interpreted here."""
    items = {}
    with ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".txt"):
                continue
            kind = _kind(name)
            if not kind:
                continue
            key = _key(name)
            items.setdefault(key, {})[kind] = name
    if not items:
        raise ValueError("No Amplitude/Phase TXT members found in ZIP")
    rows = []
    for key, members in sorted(items.items()):
        if set(members) != {"amplitude", "phase"}:
            raise ValueError(f"Incomplete amplitude/phase pair for {key!r}: {list(members)}")
        amp = members["amplitude"]
        ph = members["phase"]
        rows.append({"state": _parse_state(amp), "point": _parse_point(amp),
                     "amplitude_file": amp, "phase_file": ph})
    pairs = {(r['state'], r['point']) for r in rows}
    if len(pairs) != len(rows):
        raise ValueError('Repeated state/point mapping; edit a manual manifest')
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('w', newline='', encoding='utf8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def read_polytec(content, freq_min=1, freq_max=25000, stride=20):
    """Parse Windows-1252, seven-header-line Polytec (frequency, scalar) TXT.

    Frequencies are selected by *original line index*, preserving common 1-Hz
    frequency bins. Strict malformed rows and missing data rejection.
    """
    if isinstance(content, bytes):
        content = content.decode('cp1252')
    if stride < 1 or freq_min < 0 or freq_max < freq_min:
        raise ValueError('Invalid frequency range or stride')
    lines = content.splitlines()
    if len(lines) < 9:
        raise ValueError('Polytec export must contain a header and numeric rows')
    freq = []
    vals = []
    for i, line in enumerate(lines[7:]):
        if i % stride:
            continue
        fields = line.strip().replace(',', '.').split()
        if len(fields) < 2:
            raise ValueError(f'Malformed numeric row {i+8}')
        try:
            f, v = float(fields[0]), float(fields[1])
        except ValueError as exc:
            raise ValueError(f'Invalid numeric row {i+8}') from exc
        if freq_min <= f <= freq_max:
            freq.append(f);vals.append(v)
    if len(freq) < 5 or not np.isfinite(freq).all() or not np.isfinite(vals).all():
        raise ValueError('Too few valid finite frequency bins')
    freq = np.array(freq, dtype=np.float64)
    vals = np.array(vals, dtype=np.float64)
    if not np.all(np.diff(freq) > 0):
        raise ValueError('Non-increasing frequency axis')
    return freq, vals


def load_coordinates(csv_path):
    """Map physical point number to numeric (x,y[,z]) using CSV headers.

    Supports point, point_id, index, scan_point, point_nr, number; x,y,[z]
    and aliases x_mm,y_mm etc. Coordinates may be in mm (scale cancels in
    Gaussian kNN graph but must be consistent across all dimensions).
    """
    coords={}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames:
            raise ValueError('Missing coordinate CSV headers')
        cols={re.sub(r'[^a-z0-9]+','_',name.lower()).strip('_'):name for name in rd.fieldnames}
        # Polytec metadata may use 'X [mm]' or 'X (mm)'; normalize to x_mm.
        cols.update({re.sub(r'_(?:mm|m)$','',key):val for key,val in list(cols.items()) if re.fullmatch(r'[xyz]_(?:mm|m)',key)})
        pointkey=next((cols[k] for k in ['point','point_id','point_index','index','scan_point','point_nr','point_number','number'] if k in cols),None)
        axes=[]
        for axis in 'xy':
            k=next((cols[v] for v in [axis, axis+'_mm',axis+'_m',axis+'_coordinate'] if v in cols),None)
            if k is None:raise ValueError('CSV must have x and y coordinate columns')
            axes.append(k)
        k=next((cols[v] for v in ['z','z_mm','z_m','z_coordinate'] if v in cols),None)
        if k is not None:axes.append(k)
        if pointkey is None:
            raise ValueError('Coordinate CSV needs identifiable point index')
        for row in rd:
            p=int(row[pointkey]);xyz=np.array([float(row[k].replace(',','.')) for k in axes]);
            if p in coords or not np.isfinite(xyz).all():raise ValueError('Duplicate point or nonfinite coordinates')
            coords[p]=xyz
    if not coords:raise ValueError('No coordinates')
    return coords


def _rows(manifest_path):
    with open(manifest_path, newline='', encoding='utf8') as f:
        items=list(csv.DictReader(f))
    mandatory={'state','point','amplitude_file','phase_file'}
    if not items or not mandatory.issubset(items[0]):
        raise ValueError('Manifest requires state,point,amplitude_file,phase_file')
    seen=set()
    for row in items:
        row['point']=int(row['point'])
        key=(row['state'],row['point'])
        if key in seen:raise ValueError(f'Duplicate state/point {key}')
        seen.add(key)
    return items


def convert_archive(zip_path, manifest_csv, coordinates_csv, destination_npz,
                    *, freq_min=1, freq_max=25000, stride=20,
                    expected_states=18, expected_points=51):
    """Reconstruct raw H1 complex FRF with strict physical-point alignment.

    Takes intersection of common point indexes across states (extra 52nd in
    all-loose state is dropped); never reshuffles by independently sorted ZIP
    names. Fails rather than padding missing frequency or complex phase.
    """
    zip_path=Path(zip_path)
    coords=load_coordinates(coordinates_csv)
    rows=_rows(manifest_csv)
    states=sorted({r['state'] for r in rows})
    if len(states)!=expected_states:
        raise ValueError(f'Expected {expected_states} states, obtained {len(states)}')
    by_state={state:{r['point']:r for r in rows if r['state']==state} for state in states}
    common=set.intersection(*(set(v) for v in by_state.values()))
    common=sorted(common & set(coords))
    if len(common)!=expected_points:
        raise ValueError(f'Expected {expected_points} common points with coordinates; got {len(common)}')
    data=[]
    frequencies=None
    with ZipFile(zip_path) as archive:
        member_set=set(archive.namelist())
        for state in states:
            matrix=[]
            for point in common:
                record=by_state[state][point]
                a=record['amplitude_file'];b=record['phase_file']
                if a not in member_set or b not in member_set:
                    raise ValueError(f'Missing ZIP member for {state}, point {point}')
                af,av=read_polytec(archive.read(a),freq_min,freq_max,stride)
                pf,pv=read_polytec(archive.read(b),freq_min,freq_max,stride)
                if not np.array_equal(af,pf):
                    raise ValueError(f'Amplitude/phase frequency mismatch: {state}, {point}')
                if frequencies is None:frequencies=af
                elif not np.array_equal(frequencies,af):
                    raise ValueError(f'Frequency mismatch in state {state}, point {point}')
                if np.any(av<0):raise ValueError('Amplitude must be nonnegative (m/N; not dB)')
                matrix.append(av*np.exp(1j*np.deg2rad(pv)))
            data.append(matrix)
    h=np.asarray(data,np.complex128)
    xyz=np.stack([coords[p] for p in common])
    target=Path(destination_npz);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,frf=h,coordinates=xyz,
                        group_ids=np.asarray(states),point_ids=np.asarray(common),frequency_hz=frequencies)
    digest=hashlib.sha256()
    with zip_path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    prov={'source_doi':'10.5281/zenodo.22770267',
          'original_archive_sha256':digest.hexdigest(),
          'source':str(zip_path),'npz':str(target),'states':len(states),
          'points':len(common),'frequency_bins':len(frequencies),
          'frequency_bounds_hz':[float(frequencies[0]),float(frequencies[-1])],
          'frequency_stride':stride,'amplitude_units':'m/N',
          'phase_original_units':'degrees','complex_formula':'amplitude*exp(1j*deg2rad(phase))',
          'independence':'one complete measurement per nominal torque state; not specimen-independent',
          'data_not_redistributed':True}
    target.with_suffix('.provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    return prov
