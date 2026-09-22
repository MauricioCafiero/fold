"""Interface confidence scores (ipSAE, pDockQ, pDockQ2) for cofolding output.

ipSAE (Dunbrack 2025, "Res ipSAE loquuntur",
https://doi.org/10.1101/2025.02.10.637595) fixes AlphaFold's iptm as an
INTERFACE confidence: iptm averages over whole chains, so disordered tails
or non-interacting domains drag it down. ipSAE keeps only residue pairs with
interchain PAE below a cutoff and rescales the TM-score d0 by how many such
pairs exist -- a "mini-chain" of just the interface. It separates true from
false complexes better than iptm and, unlike iptm, is insensitive to how much
of each chain actually touches the interface.

The score core is adapted from ColabFold's vendored implementation
(colabfold/alphafold/ipsae.py, MIT -- itself a reimplementation of
DunbrackLab/IPSAE), trimmed of alphafold-package imports: this version
operates directly on an AF3-style full confidences JSON (pae matrix +
token_chain_ids/token_res_ids) plus the mmCIF the JSON was written alongside,
as produced by RF3 and (post image update, see below) OpenFold3.

NOTE: the paper validated ipSAE on protein-protein heterodimers. Protein-
LIGAND interfaces are computed the same way here (the ligand is just a small
pseudo-chain of atom tokens) but are NOT benchmarked by the paper -- treat
protein-ligand ipSAE values as experimental.

Requires the 'pae' matrix, so it needs a full *_confidences.json. Our stored
OpenFold3 smoke-test output only has 'pde' -- the openfoldconsortium/openfold3
Docker tag we ran predates openfold3 PR #142, which added the PAE matrix to
full confidence output; the PyPI openfold3 package (Colab path) includes it.
This module raises a targeted error in that case rather than substituting
pde, which is a different quantity (predicted DISTANCE error).
"""

import json
import re
from pathlib import Path

import numpy as np

DEFAULT_PAE_CUTOFF = 15.0
PDOCKQ_DIST_CUTOFF = 8.0


def _ptm_score(pae, d0):
    """TM-score kernel."""
    return 1.0 / (1.0 + (pae / d0) ** 2.0)


def _calc_d0_array(n_res):
    """TM-score d0 (Yang & Skolnick 2004), minimum 1.0."""
    length = np.maximum(26.0, np.asarray(n_res, dtype=np.float64))
    return np.maximum(1.0, 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8)


def compute_interface_scores(
    pae: np.ndarray,
    token_chain_ids: np.ndarray,
    token_coords: np.ndarray,
    token_plddts: np.ndarray,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
) -> dict:
    """Per-chain-pair interface scores from token-level data.

    pae: (n_tokens, n_tokens) predicted aligned error in Angstrom.
    token_chain_ids: (n_tokens,) chain label per token (e.g. "A_1", "B_1").
    token_coords: (n_tokens, 3) representative coordinate per token (CB of
        the residue, or the atom itself for ligand atom tokens).
    token_plddts: (n_tokens,) per-token pLDDT (CA pLDDT for protein residues).

    Returns {"ipsae": {...}, "pdockq": {...}, "pdockq2": {...}} keyed by
    "X-Y" chain pairs (ipSAE in both directions, pDockQ/pDockQ2 symmetric).
    """
    token_chain_ids = np.asarray(token_chain_ids).astype(str)
    pae = np.asarray(pae, dtype=np.float64)
    token_coords = np.asarray(token_coords, dtype=np.float64)
    token_plddts = np.asarray(token_plddts, dtype=np.float64)

    unique_chains = list(dict.fromkeys(token_chain_ids))
    if len(unique_chains) < 2:
        raise ValueError("need at least two chains for interface scores")

    # CB-style distance matrix from the representative coordinates.
    deltas = token_coords[:, None, :] - token_coords[None, :, :]
    distances = np.sqrt((deltas**2).sum(-1))

    ipsae, pdockq, pdockq2 = {}, {}, {}
    for chain_1 in unique_chains:
        for chain_2 in unique_chains:
            if chain_1 == chain_2:
                continue
            key = f"{chain_1}-{chain_2}"
            pair_mask = np.outer(token_chain_ids == chain_1,
                                 token_chain_ids == chain_2)

            # ipSAE: d0 per residue row, from that row's count of interchain
            # pairs with good PAE (the paper's "d0res" variant).
            valid_pairs = pair_mask & (pae < pae_cutoff)
            n0_res = valid_pairs.sum(axis=1)
            d0_res = _calc_d0_array(n0_res)
            ptm_sums = (_ptm_score(pae, d0_res[:, None]) * valid_pairs).sum(axis=1)
            ipsae_by_res = np.divide(ptm_sums, n0_res,
                                     out=np.zeros_like(ptm_sums), where=n0_res > 0)
            ipsae[key] = round(float(ipsae_by_res.max()), 6)

            # pDockQ / pDockQ2 from interface contacts + pLDDT.
            contacts = pair_mask & (distances <= PDOCKQ_DIST_CUTOFF)
            n_contacts = int(contacts.sum())
            if n_contacts > 0:
                interface_res = contacts.any(axis=1) | contacts.any(axis=0)
                mean_plddt = token_plddts[interface_res].mean()
                pdockq_val = 0.724 / (1.0 + np.exp(
                    -0.052 * (mean_plddt * np.log10(n_contacts) - 152.611))) + 0.018
                mean_ptm = _ptm_score(pae[contacts], 10.0).mean()
                pdockq2_val = 1.31 / (1.0 + np.exp(
                    -0.075 * (mean_plddt * mean_ptm - 84.733))) + 0.005
            else:
                pdockq_val, pdockq2_val = 0.0, 0.0
            if chain_1 < chain_2:  # symmetric metrics, keep one direction
                pdockq[key] = round(float(pdockq_val), 4)
            pdockq2[key] = round(float(pdockq2_val), 4)

    return {"ipsae": ipsae, "pdockq": pdockq, "pdockq2": pdockq2}


def format_interface_scores(scores: dict) -> str:
    """One-line summary: best ipSAE per interface (direction-independent)."""
    pair_max = {}
    for key, value in scores["ipsae"].items():
        chain_1, chain_2 = key.split("-")
        pair = key if chain_1 < chain_2 else f"{chain_2}-{chain_1}"
        pair_max[pair] = max(pair_max.get(pair, 0.0), value)
    return ",".join(f"{pair}:{value:.3g}"
                    for pair, value in sorted(pair_max.items()))


def _token_base_chain(token_chain_id: str) -> str:
    """RF3 labels token chains "A_1" while the CIF chain is "A"."""
    return re.sub(r"_\d+$", "", token_chain_id)


def locate_conf_json(cif_path: str | Path) -> Path:
    """Find the full confidences JSON written next to a model CIF.

    Same naming convention results.find_best_cif relies on: the CIF
    <name>_model.cif (optionally with a seed/sample stem) sits beside
    <name>_confidences.json.
    """
    cif_path = Path(cif_path)
    stem = cif_path.stem
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    return cif_path.with_name(f"{stem}_confidences.json")


def _derive_af3_tokens(
    structure: "gemmi.Structure", n_tokens: int, atom_plddt_raw
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build token arrays from the CIF when the confidences JSON has none.

    OpenFold3 (AF3-format) full confidences carry no per-token arrays --
    only plddt/pde/pae, where plddt is per-ATOM pLDDT in CIF file order
    (1689 = 962 + 727 atoms for the barnase-barstar dimer). Tokens follow
    the AF3 convention: one per protein residue (its CA atom) or per
    ligand atom, in CIF file order.
    """
    import gemmi

    atom_plddts = np.asarray(atom_plddt_raw, dtype=np.float64)
    token_chain_ids: list[str] = []
    token_res_ids: list[int] = []
    n_atoms_total = 0
    for chain in structure[0]:
        # Water chains carry no tokens in AF3 output.
        residues = [res for res in chain
                    if not (gemmi.find_tabulated_residue(res.name) or
                            gemmi.ResidueInfo()).is_water()
                    if res.name != "HOH"]
        ca_count = sum(1 for res in residues if any(a.name == "CA" for a in res))
        if ca_count and ca_count == len(residues):
            # Protein-style: one token per residue.
            for res in residues:
                token_chain_ids.append(chain.name)
                token_res_ids.append(res.seqid.num - 1)
        else:
            # Ligand-style: one token per atom.
            for res in residues:
                for _ in res:
                    token_chain_ids.append(chain.name)
                    token_res_ids.append(res.seqid.num - 1)
        n_atoms_total += sum(len(res) for res in chain)

    if len(token_chain_ids) != n_tokens:
        raise ValueError(
            f"CIF tokenization gives {len(token_chain_ids)} tokens "
            f"(protein residues + ligand atoms) but the PAE matrix is "
            f"{n_tokens} -- unsupported chain composition"
        )
    if len(atom_plddts) != n_atoms_total:
        raise ValueError(
            f"plddt vector has {len(atom_plddts)} entries but the CIF has "
            f"{n_atoms_total} atoms -- cannot align per-atom pLDDT"
        )
    return (
        np.asarray(token_chain_ids).astype(str),
        np.asarray(token_res_ids),
        atom_plddts,
        np.asarray([], dtype=str),  # no atom_chain_ids to cross-check
    )


def interface_scores(
    conf_json_path: str | Path,
    cif_path: str | Path,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
) -> dict:
    """Compute per-interface scores for one model's confidences JSON + CIF.

    Returns the dict from compute_interface_scores plus a "summary" string.
    """
    import gemmi

    conf = json.loads(Path(conf_json_path).read_text())
    if "pae" not in conf:
        raise ValueError(
            f"no PAE matrix in {conf_json_path} -- this output predates "
            "openfold3 PR #142 (PAE in full confidence output); re-run with "
            "an updated openfold3 Docker image / PyPI package. (The matrix "
            "present is 'pde', predicted distance error, which ipSAE cannot "
            "use.)"
        )

    pae = np.asarray(conf["pae"], dtype=np.float64)
    n_tokens = pae.shape[0]

    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()

    if "token_chain_ids" in conf:
        # RF3: the JSON carries per-token chain/res arrays directly.
        token_chain_ids = np.asarray(conf["token_chain_ids"]).astype(str)
        token_res_ids = np.asarray(conf["token_res_ids"])
        atom_plddts = np.asarray(conf.get("atom_plddts", []), dtype=np.float64)
        atom_chain_ids = np.asarray(conf.get("atom_chain_ids", [])).astype(str)
    else:
        # OpenFold3 / AF3-format full confidences: only pldt/pde/pae, with
        # plddt per-ATOM in CIF file order -- derive tokens from the CIF.
        (token_chain_ids, token_res_ids,
         atom_plddts, atom_chain_ids) = _derive_af3_tokens(
            structure, n_tokens, conf.get("plddt", [])
        )
    if len(token_chain_ids) != n_tokens or len(token_res_ids) != n_tokens:
        raise ValueError(
            f"token arrays (chain {len(token_chain_ids)}, res {len(token_res_ids)}) "
            f"don't match the PAE matrix size {n_tokens}"
        )

    # The JSON's atom arrays (atom_chain_ids / atom_plddts) are in CIF file
    # order -- verified against real RF3 output. Flatten the CIF atoms of
    # every chain the tokens reference, in that same order, and cross-check.
    token_chains = list(dict.fromkeys(token_chain_ids))
    flat: list[tuple] = []  # (chain, atom_name, pos, residue, plddt) in file order
    chain_offsets: dict[str, int] = {}
    chain_atom_counts: dict[str, int] = {}
    for tc in token_chains:
        chain_name = _token_base_chain(tc)
        chain_model = next((c for m in structure for c in m if c.name == chain_name), None)
        if chain_model is None:
            raise ValueError(f"token chain {tc!r} has no CIF chain {chain_name!r}")
        chain_offsets[tc] = len(flat)
        n_chain_tokens = int((token_chain_ids == tc).sum())
        for residue in chain_model:
            for atom in residue:
                flat.append((chain_name, atom.name, atom.pos, residue, 0.0))
        n_atoms = len(flat) - chain_offsets[tc]
        chain_atom_counts[tc] = n_atoms
        if n_atoms == n_chain_tokens:
            # Ligand-style: one token per atom -> per-token pLDDT is that
            # atom's pLDDT.
            for k in range(chain_offsets[tc], len(flat)):
                flat[k] = flat[k][:4] + (float(atom_plddts[k]),)
        elif n_atoms > n_chain_tokens:
            # Protein-style: one token per residue; per-token pLDDT is the
            # CA atom's pLDDT (the AlphaFold convention for residue pLDDT).
            for k in range(chain_offsets[tc], len(flat)):
                if flat[k][1] == "CA":
                    flat[k] = flat[k][:4] + (float(atom_plddts[k]),)
        else:
            raise ValueError(
                f"chain {chain_name!r}: {n_chain_tokens} tokens but only "
                f"{n_atoms} CIF atoms -- token/atom mismatch"
            )

    if len(atom_chain_ids) and len(atom_plddts) == len(atom_chain_ids):
        # The JSON's atom chain letters must follow the same file order.
        expected = [entry[0] for entry in flat]
        if list(atom_chain_ids) != expected:
            raise ValueError(
                "confidences JSON atom order doesn't match CIF file order "
                f"(expected {expected[:4]}, got {list(atom_chain_ids)[:4]})"
            )

    token_coords = np.zeros((n_tokens, 3))
    token_plddts = np.zeros(n_tokens)
    for tc in token_chains:
        chain_name = _token_base_chain(tc)
        offset, n_atoms = chain_offsets[tc], chain_atom_counts[tc]
        chain_atoms = flat[offset : offset + n_atoms]
        token_idxs = np.where(token_chain_ids == tc)[0]
        if len(token_idxs) == n_atoms:
            # Ligand-style: j-th token of this chain = j-th CIF atom.
            for j, i in enumerate(token_idxs):
                _, _, pos, _, plddt = chain_atoms[j]
                token_coords[i] = (pos.x, pos.y, pos.z)
                token_plddts[i] = plddt
        else:
            # Protein-style: one token per residue, res_id 0-based -> gemmi
            # seqid is 1-based. Representative coordinate is CB when present
            # (matching the reference implementation), else CA.
            for i in token_idxs:
                seqid = int(token_res_ids[i]) + 1
                res_atoms = [e for e in chain_atoms if e[3].seqid.num == seqid]
                if not res_atoms:
                    raise ValueError(
                        f"no CIF residue at seqid {seqid} for token chain {tc!r}"
                    )
                atom = next((e for e in res_atoms if e[1] == "CB"), None) \
                    or next((e for e in res_atoms if e[1] == "CA"), None)
                ca = next((e for e in res_atoms if e[1] == "CA"), None)
                if atom is None or ca is None:
                    raise ValueError(
                        f"CIF residue at seqid {seqid} in chain {chain_name!r} "
                        "has no CA atom"
                    )
                token_coords[i] = (atom[2].x, atom[2].y, atom[2].z)
                token_plddts[i] = ca[4]

    # pDockQ/pDockQ2's sigmoids are calibrated on AF2/AF3 pLDDT on a 0-100
    # scale, but RF3 writes atom_plddts on a 0-1 scale (verified against the
    # smoke-test output: ligand atom pLDDT 0.8 for a confident pose). Detect
    # and rescale so the metrics aren't pinned at their floor. ipSAE itself
    # doesn't use pLDDT and is unaffected.
    if token_plddts.size and token_plddts.max() <= 1.0:
        token_plddts *= 100.0

    scores = compute_interface_scores(
        pae, token_chain_ids, token_coords, token_plddts, pae_cutoff
    )
    # Report with the CIF's chain letters ("A-B"), not token ids ("A_1-B_1").
    scores = {
        metric: {_relabel(k): v for k, v in table.items()}
        for metric, table in scores.items()
    }
    scores["summary"] = format_interface_scores(scores)
    return scores


def _relabel(key: str) -> str:
    chain_1, chain_2 = key.split("-")
    return f"{_token_base_chain(chain_1)}-{_token_base_chain(chain_2)}"