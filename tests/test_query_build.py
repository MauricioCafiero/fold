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


def test_rf3_single_component_is_apo():
    """rf3_core.build_query_components returns a no-ligand component list when
    the SMILES is empty -- the same apo relaxation as OF3's."""
    from fold.rf3_core import build_query_components

    assert build_query_components("MLS", "") == [{"seq": "MLS", "chain_id": "A"}]
    comps = build_query_components("MLS", "COc1ccc(C(=O)c2ccccc2)c(O)c1")
    assert [c["chain_id"] if "chain_id" in c else "ligand" for c in comps] == ["A", "ligand"]

def test_pocket_constraint_shape():
    """OpenFold3's schema: residue-level [chain_id, residue_id] pairs and a
    ligand CHAIN (no ligand atom), so a Boltz contact hint's token2 is
    dropped. Present since openfold3 0.4.5; the pinned 0.5.0 takes it."""
    from fold.openfold3_core import build_pocket_constraint

    assert build_pocket_constraint((4, 24), 4.2) == {
        "ligand_chain_id": "Z",
        "pocket_residues": [["A", 4], ["A", 24]],
        "max_distance": 4.2,
    }


def test_pocket_constraint_default_distance():
    from fold.openfold3_core import build_pocket_constraint

    assert build_pocket_constraint([7])["max_distance"] == 4.0
