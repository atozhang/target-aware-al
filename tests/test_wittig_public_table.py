"""Schema and identifier locks for the released Wittig tables.

`data/Wittig/wittig_literature_f119.csv` (734 literature records) and
`data/Wittig/wittig_cas_f119.csv` (233 CAS-derived records) together form the
967-reaction Wittig dataset, so their shapes, stable record IDs, and the
source-entry structure of the literature table are pinned. Both F119 blocks
are checked against `feature_config.py`, the schema the released analyses use.

`source_table` + `source_entry` is the literature-source identifier; it is not
unique on its own, because a few source entries report more than one
experimental condition. `condition_index` separates those.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "data" / "Wittig" / "wittig_literature_f119.csv"
CAS_TABLE = REPO / "data" / "Wittig" / "wittig_cas_f119.csv"

_WITTIG_DIR = REPO / "data" / "Wittig"
if str(_WITTIG_DIR) not in sys.path:
    sys.path.insert(0, str(_WITTIG_DIR))
from feature_config import STANDARDIZED_FEATURES  # noqa: E402

N_RECORDS = 734
N_SOURCE_PAIRS = 729
REPEATED_PAIRS = {(19, 78): 3, (19, 79): 3, (20, 5): 2}
SOURCE_TABLES = {11, 13, 14, 15, 16, 17, 18, 19, 20, 21}
METADATA_COLUMNS = [
    "record_id",
    "source_table",
    "source_entry",
    "condition_index",
    "z_selectivity_percent",
    "temperature_c",
    "solvent_reported",
    "solvent_standardized",
    "solvent_smiles",
    "aldehyde_smiles",
    "ylide_smiles",
    "base_smiles",
    "in_situ",
    "rxn_smiles",
]
RECORD_ID = re.compile(r"^VP1994_T\d{2}_E\d{3}_C\d{2}$")
N_CAS_RECORDS = 233
CAS_METADATA_COLUMNS = [
    "record_id", "dataset_version", "source_dataset", "z_selectivity_percent",
    "rxn_smiles", "aldehyde_smiles", "ylide_smiles", "base_smiles",
    "cis_opa_smiles", "trans_opa_smiles", "in_situ", "temperature_c",
    "cas_n_stages", "cas_olefination_stage", "cas_stage_conditions",
    "solvent_record_status", "solvent_reported", "solvent_cas_rn",
    "solvent_smiles", "solvent_stage", "solvent_other_stages", "cas_stage_trace",
]
CAS_RECORD_ID = re.compile(r"^(CAS_R\d{3}|X48_\d{2})$")


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    # keep_default_na=False: `base_smiles` carries the literal string "NA" as a
    # source value, which the default reader would turn into a missing value.
    return pd.read_csv(TABLE, keep_default_na=False, na_filter=False)


def test_shape_is_734_by_133(table):
    assert table.shape == (N_RECORDS, len(METADATA_COLUMNS) + 119)


def test_metadata_columns_come_first_in_order(table):
    assert list(table.columns[: len(METADATA_COLUMNS)]) == METADATA_COLUMNS


def test_f119_block_matches_feature_config(table):
    assert list(table.columns[len(METADATA_COLUMNS) :]) == STANDARDIZED_FEATURES


def test_descriptor_columns_are_numeric_and_complete(table):
    block = table[STANDARDIZED_FEATURES].apply(pd.to_numeric, errors="coerce")
    assert not block.isna().to_numpy().any()


def test_record_id_is_unique_and_well_formed(table):
    ids = table["record_id"]
    assert ids.nunique() == N_RECORDS
    assert ids.map(lambda s: bool(RECORD_ID.match(s))).all()


def test_record_id_is_built_from_its_own_source_columns(table):
    expected = [
        f"VP1994_T{int(t):02d}_E{int(e):03d}_C{int(c):02d}"
        for t, e, c in zip(
            table["source_table"], table["source_entry"], table["condition_index"]
        )
    ]
    assert list(table["record_id"]) == expected


def test_source_pair_multiplicities(table):
    pairs = Counter(
        zip(table["source_table"].astype(int), table["source_entry"].astype(int))
    )
    assert len(pairs) == N_SOURCE_PAIRS
    assert {p: n for p, n in pairs.items() if n > 1} == REPEATED_PAIRS
    assert sum(pairs.values()) == N_RECORDS


def test_condition_index_starts_at_one_with_no_gaps(table):
    seen = defaultdict(list)
    for t, e, c in zip(
        table["source_table"].astype(int),
        table["source_entry"].astype(int),
        table["condition_index"].astype(int),
    ):
        seen[(t, e)].append(c)
    assert all(v == list(range(1, len(v) + 1)) for v in seen.values())


def test_source_tables_are_the_expected_literature_tables(table):
    assert set(table["source_table"].astype(int)) == SOURCE_TABLES


def test_no_working_data_or_private_fields_are_released(table):
    banned_exact = {
        "table", "entry", "z", "temp", "insitu", "Solvent", "solvent_name",
        "X", "pdt_smiles", "reactants", "spectators", "cis_pdt", "trans_pdt",
        "cis_mapped_rxn", "trans_mapped_rxn", "ald_id", "ylide_id", "cis_id",
        "trans_id", "base_id", "solvent_id",
    }
    banned_substrings = ("xyz", "thermo", "mapped", "cas", "pool", "task", "price")
    for col in table.columns:
        assert col not in banned_exact, col
        assert not any(s in col.lower() for s in banned_substrings), col


def test_base_smiles_na_is_a_source_value_not_a_blank(table):
    values = table["base_smiles"]
    assert (values == "NA").any()
    assert values.map(lambda s: bool(str(s).strip())).all()


def test_reported_and_standardized_solvent_are_both_populated(table):
    for col in ("solvent_reported", "solvent_standardized"):
        assert table[col].map(lambda s: bool(str(s).strip())).all()
    # the reported column keeps the transcribed notation, so it is the coarser
    # of the two labels only after standardization collapses variants
    assert table["solvent_reported"].nunique() >= table["solvent_standardized"].nunique()


def test_z_selectivity_is_a_percentage(table):
    z = pd.to_numeric(table["z_selectivity_percent"])
    assert z.notna().all()
    assert z.between(0, 100).all()


def test_rows_are_rectangular_on_disk(table):
    with TABLE.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == N_RECORDS + 1
    assert all(len(r) == len(rows[0]) for r in rows)


@pytest.fixture(scope="module")
def cas_table() -> pd.DataFrame:
    return pd.read_csv(CAS_TABLE, keep_default_na=False, na_filter=False)


def test_cas_table_shape_and_column_order(cas_table):
    assert cas_table.shape == (N_CAS_RECORDS, len(CAS_METADATA_COLUMNS) + 119)
    assert list(cas_table.columns[: len(CAS_METADATA_COLUMNS)]) == CAS_METADATA_COLUMNS
    assert list(cas_table.columns[len(CAS_METADATA_COLUMNS) :]) == STANDARDIZED_FEATURES


def test_cas_records_are_identified_complete_and_labeled(cas_table):
    assert cas_table["record_id"].is_unique
    assert cas_table["record_id"].map(lambda s: bool(CAS_RECORD_ID.match(s))).all()
    assert set(cas_table["source_dataset"]) == {"cas"}
    block = cas_table[STANDARDIZED_FEATURES].apply(pd.to_numeric, errors="coerce")
    assert not block.isna().to_numpy().any()
    assert pd.to_numeric(cas_table["z_selectivity_percent"]).between(0, 100).all()
    for col in ("rxn_smiles", "aldehyde_smiles", "ylide_smiles", "cis_opa_smiles", "trans_opa_smiles"):
        assert cas_table[col].map(lambda s: bool(s.strip()) and s != "NA").all(), col


def test_the_two_tables_form_the_967_reaction_dataset(table, cas_table):
    assert len(table) + len(cas_table) == 967
    assert not set(table["record_id"]) & set(cas_table["record_id"])
