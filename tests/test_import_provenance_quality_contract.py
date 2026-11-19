"""Synthetic import contracts: source provenance and train-only coherence.

The bundled example is generated data, not an experimental validation set.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparseldv_qa.user_data import convert


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "step3_user_data"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_import_manifest_hashes_match_source_schema_and_output(tmp_path):
    source = FIXTURES / "frf_long.csv"
    schema = FIXTURES / "schema.json"
    output = tmp_path / "converted.npz"

    report = convert(source, schema, output)
    manifest = json.loads(output.with_suffix(".npz.manifest.json").read_text(encoding="utf-8"))

    assert manifest == report
    assert report["source_sha256"] == _digest(source)
    assert report["mapping_sha256"] == _digest(schema)
    assert report["output_sha256"] == _digest(output)
    assert report["algorithm_execution"] is False
    assert report["input_groups_are_independent"] == "not established by import"


def test_holdout_coherence_cannot_change_training_quality(tmp_path):
    source = FIXTURES / "frf_long.csv"
    schema = FIXTURES / "schema.json"
    original = tmp_path / "original.npz"
    convert(source, schema, original)

    table = pd.read_csv(source)
    holdout = table["scan"] == "holdout_C"
    assert holdout.any()
    table.loc[holdout, "coherence"] = 0.0
    modified_source = tmp_path / "modified.csv"
    table.to_csv(modified_source, index=False)
    modified = tmp_path / "modified.npz"
    changed_report = convert(modified_source, schema, modified)

    with np.load(original, allow_pickle=False) as before, np.load(
        modified, allow_pickle=False
    ) as after:
        np.testing.assert_array_equal(before["quality"], after["quality"])
        np.testing.assert_array_equal(before["training"], after["training"])
        np.testing.assert_array_equal(before["truth"], after["truth"])
    assert changed_report["quality_computed_from_training_only"] is True
    assert changed_report["source_sha256"] == _digest(modified_source)
    assert changed_report["source_sha256"] != _digest(source)
