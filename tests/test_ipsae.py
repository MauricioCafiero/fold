import json

import numpy as np
import pytest

from fold.ipsae import (
    compute_interface_scores,
    format_interface_scores,
    interface_scores,
    locate_conf_json,
)
from fold.analyze import report_interface_scores

# 5 tokens: 3 protein residues ("A") + 2 ligand atom tokens ("B"). The core
# takes any chain labels; the "A_1"-style token ids are stripped by the glue.
CHAINS = np.array(["A", "A", "A", "B", "B"])
# CB/atom coordinates along z: everything within 8 A of the ligand atoms,
# so all 3x2 interchain pairs count as contacts.
COORDS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 3.8],
    [0.0, 0.0, 7.6],
    [0.0, 0.0, 0.5],
    [0.0, 0.0, 4.3],
])
PLDDTS = np.full(5, 90.0)


def _pae(interchain, n=5, n_prot=3):
    pae = np.full((n, n), 1.0)  # intrachain: excellent
    pae[:n_prot, n_prot:] = interchain
    pae[n_prot:, :n_prot] = interchain
    return pae


def test_perfect_interface_gives_ipsae_one():
    scores = compute_interface_scores(
        _pae(0.0), CHAINS, COORDS, PLDDTS
    )
    assert scores["ipsae"]["A-B"] == 1.0
    assert scores["ipsae"]["B-A"] == 1.0


def test_no_good_pairs_gives_ipsae_zero():
    # Interchain PAE everywhere above the 15 A cutoff -> no valid pairs.
    scores = compute_interface_scores(
        _pae(50.0), CHAINS, COORDS, PLDDTS
    )
    assert scores["ipsae"]["A-B"] == 0.0
    assert scores["ipsae"]["B-A"] == 0.0


def test_minichain_d0_scaling():
    # Same interchain PAE (2 A) but a 300-residue partner vs a 30-residue
    # one: the bigger interface mini-chain gets a larger d0, so the same
    # PAE yields a higher ipSAE. Hand-computed: d0(30)=1.24*15^(1/3)-1.8,
    # d0(300)=1.24*285^(1/3)-1.8.
    def chain_of(n, label, start):
        chains = np.full(n, label)
        return chains

    x_chains = np.full(300, "X")
    y_chains = np.full(30, "Y")
    chains = np.concatenate([x_chains, y_chains])
    coords = np.random.default_rng(0).uniform(0, 100, (330, 3))
    pae = np.full((330, 330), 2.0)
    pae[:300, 300:] = 2.0
    pae[300:, :300] = 2.0

    scores = compute_interface_scores(pae, chains, coords, np.full(330, 90.0))
    d0_small = 1.24 * (30 - 15) ** (1 / 3) - 1.8  # rows of X: n0 = |Y| = 30
    d0_large = 1.24 * (300 - 15) ** (1 / 3) - 1.8  # rows of Y: n0 = |X| = 300
    expected_small = 1 / (1 + (2 / d0_small) ** 2)
    expected_large = 1 / (1 + (2 / d0_large) ** 2)
    assert scores["ipsae"]["X-Y"] == pytest.approx(expected_small, abs=1e-6)
    assert scores["ipsae"]["Y-X"] == pytest.approx(expected_large, abs=1e-6)
    assert scores["ipsae"]["Y-X"] > scores["ipsae"]["X-Y"]


def test_format_interface_scores_takes_max_over_directions():
    scores = {"ipsae": {"A-B": 0.13, "B-A": 0.26}, "pdockq": {}, "pdockq2": {}}
    assert format_interface_scores(scores) == "A-B:0.26"


# Synthetic structure built with gemmi (a hand-written bare atom_site loop
# is too minimal for gemmi's mmCIF reader): 3 protein residues on chain A,
# each with N/CA/C/CB, plus a 2-atom ligand on chain B.
PROTEIN_RESIDUES = [("GLY", 1.0), ("ALA", 4.8), ("LEU", 8.6)]  # (name, CA x)
LIGAND_ATOMS = [("C0", (0.3, 0.1, 0.2)), ("N0", (0.4, 0.2, 0.3))]


def _write_cif(tmp_path):
    import gemmi

    structure = gemmi.Structure()
    structure.name = "test"
    model = gemmi.Model("1")
    protein = gemmi.Chain("A")
    for i, (resname, ca_x) in enumerate(PROTEIN_RESIDUES):
        residue = gemmi.Residue()
        residue.name = resname
        residue.seqid = gemmi.SeqId(i + 1, " ")
        residue.het_flag = "A"
        for name, pos in [("N", (ca_x - 1.0, 0.0, 0.0)), ("CA", (ca_x, 0.0, 0.0)),
                          ("C", (ca_x + 1.0, 0.0, 0.0)), ("CB", (ca_x, 1.0, 0.0))]:
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element("C" if name.startswith("C") else "N")
            atom.pos = gemmi.Position(*pos)
            residue.add_atom(atom)
        protein.add_residue(residue)
    ligand = gemmi.Chain("B")
    lig_residue = gemmi.Residue()
    lig_residue.name = "LIG"
    lig_residue.seqid = gemmi.SeqId(1, " ")
    lig_residue.het_flag = "H"
    for name, pos in LIGAND_ATOMS:
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element(name[0])
        atom.pos = gemmi.Position(*pos)
        lig_residue.add_atom(atom)
    ligand.add_residue(lig_residue)
    model.add_chain(protein)
    model.add_chain(ligand)
    structure.add_model(model)
    structure.setup_entities()
    path = tmp_path / "job_model.cif"
    path.write_text(structure.make_mmcif_document().as_string())
    return path


def _write_output(tmp_path):
    cif = _write_cif(tmp_path)
    # 5 tokens: 3 protein residues + 2 ligand atoms; ligand atom tokens get
    # unique res_ids continuing after the protein's (RF3 convention).
    pae = _pae(2.0)
    conf = {
        "pae": pae.tolist(),
        "token_chain_ids": ["A_1", "A_1", "A_1", "B_1", "B_1"],
        "token_res_ids": [0, 1, 2, 3, 4],
        "atom_chain_ids": ["A"] * 12 + ["B"] * 2,
        # pLDDT on RF3's 0-1 scale: exercises the rescale to 0-100.
        "atom_plddts": [0.9] * 12 + [0.8, 0.8],
    }
    conf_path = tmp_path / "job_confidences.json"
    conf_path.write_text(json.dumps(conf))
    return conf_path, cif


def test_interface_scores_glue(tmp_path):
    conf_path, cif = _write_output(tmp_path)

    scores = interface_scores(conf_path, cif)

    # Rows of A see n0=2 good pairs -> d0 floor 1.0 -> kernel(2.0, 1.0)=0.2.
    assert scores["ipsae"]["A-B"] == pytest.approx(0.2, abs=1e-6)
    # Rows of B see n0=3 good pairs -> length floor 26 -> d0 floor 1.0.
    assert scores["ipsae"]["B-A"] == pytest.approx(0.2, abs=1e-6)
    # Contacts exist and pLDDT was rescaled to the 0-100 scale. With only 4
    # contacts pDockQ stays small (its sigmoid wants tens of contacts) but
    # must be above its 0.018 floor, and pDockQ2 lands ~0.59.
    assert scores["pdockq"]["A-B"] > 0.018
    assert scores["pdockq2"]["A-B"] == pytest.approx(0.587, abs=0.01)
    # Chain labels are reported with the CIF's chain letters.
    assert scores["summary"] == "A-B:0.2"


def test_interface_scores_af3_fallback(tmp_path):
    """OpenFold3/AF3 full confidences have no token arrays -- only a
    per-ATOM plddt vector in CIF file order. The glue must derive tokens
    (one per protein residue, one per ligand atom) from the CIF."""
    cif = _write_cif(tmp_path)
    # 14 atoms in CIF file order: 12 protein + 2 ligand. Token count is
    # 3 residues + 2 ligand atoms = 5, matching the PAE matrix.
    conf = {
        "pae": _pae(2.0).tolist(),
        # pLDDT on a 0-1 scale: exercises the rescale to 0-100.
        "plddt": [0.9] * 12 + [0.8, 0.8],
    }
    conf_path = tmp_path / "job_confidences.json"
    conf_path.write_text(json.dumps(conf))

    scores = interface_scores(conf_path, cif)

    # Same hand-checkable arithmetic as the RF3 glue test: the token
    # pLDDTs and coordinates land identically, so all metrics must too.
    assert scores["ipsae"]["A-B"] == pytest.approx(0.2, abs=1e-6)
    assert scores["ipsae"]["B-A"] == pytest.approx(0.2, abs=1e-6)
    assert scores["pdockq"]["A-B"] > 0.018
    assert scores["pdockq2"]["A-B"] == pytest.approx(0.587, abs=0.01)
    assert scores["summary"] == "A-B:0.2"


def test_interface_scores_missing_pae_raises(tmp_path):
    cif = _write_cif(tmp_path)
    conf_path = tmp_path / "job_confidences.json"
    conf_path.write_text(json.dumps({"pde": np.zeros((5, 5)).tolist(),
                                     "token_chain_ids": ["A_1"] * 5}))

    with pytest.raises(ValueError, match="PR #142"):
        interface_scores(conf_path, cif)


def test_locate_conf_json(tmp_path):
    cif = tmp_path / "job_seed-0_sample-1_model.cif"
    assert locate_conf_json(cif) == cif.with_name("job_seed-0_sample-1_confidences.json")
    plain = tmp_path / "model.cif"
    assert locate_conf_json(plain) == plain.with_name("model_confidences.json")


def test_report_interface_scores_prints_summary(tmp_path, capsys):
    conf_path, cif = _write_output(tmp_path)
    # find_best_cif picks by the aggregated confidences file the cofolding
    # tools write alongside each sample.
    (tmp_path / "job_confidences_aggregated.json").write_text(
        json.dumps({"sample_ranking_score": 1.0})
    )

    report_interface_scores(tmp_path)

    out = capsys.readouterr().out
    assert "job_model.cif" in out
    assert "interface A-B: ipSAE 0.200" in out


def test_report_interface_scores_is_best_effort(tmp_path, capsys):
    """A cofold whose confidences can't be scored must not fail the run."""
    empty = tmp_path / "no_outputs"
    empty.mkdir()

    report_interface_scores(empty)

    out = capsys.readouterr().out
    assert "unavailable" in out