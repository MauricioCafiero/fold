"""Query-chain construction for OpenFold3: protein+ligand, dimer, both."""

import pytest

from fold.openfold3_core import build_query_chains


def test_protein_plus_ligand():
    chains = build_query_chains("MLS", "CCO")
    assert chains == [
        {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "MLS"},
        {"molecule_type": "ligand", "chain_ids": ["Z"], "smiles": "CCO"},
    ]


def test_dimer_two_protein_chains_no_ligand():
    chains = build_query_chains("MLS", "", second_sequence="TET")
    assert chains == [
        {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "MLS"},
        {"molecule_type": "protein", "chain_ids": ["B"], "sequence": "TET"},
    ]


def test_dimer_plus_ligand():
    chains = build_query_chains("MLS", "CCO", second_sequence="TET")
    assert [c["chain_ids"][0] for c in chains] == ["A", "B", "Z"]
    assert [c["molecule_type"] for c in chains] == ["protein", "protein", "ligand"]


def test_single_chain_is_apo():
    """A lone protein chain builds a valid apo query (no ligand, no partner);
    the old >=2-chains guard was removed 2026-10-09."""
    chains = build_query_chains("MLS", "")
    assert [c["chain_ids"] for c in chains] == [["A"]]