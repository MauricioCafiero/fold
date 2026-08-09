"""AutoDock Vina docking helpers, adapted from MauricioCafiero/dock_assist
(code/vina_dock.py), trimmed to the "dock at a known site" path this repo
needs -- no blind pocket detection, no LLM-agent tool wrappers. Copied in
rather than imported from a sibling directory so this repo has no
dependency on dock_assist's presence or layout on disk.

Needs its own optional dependencies -- see pyproject.toml's
[project.optional-dependencies] "docking" group; run
`uv sync --extra docking` before using this module. Also needs the
`obabel` CLI (Open Babel) on PATH, provided by the openbabel-wheel package.
"""

import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace


class DockError(Exception):
    """Recoverable docking failure (bad SMILES, Vina crash, obabel error, ...)."""


def find_vina_bin(explicit=None):
    """Locate the Vina binary: an explicit path, else dockstring's vendored copy."""
    if explicit:
        if not os.path.exists(explicit):
            raise DockError(f"--vina-bin not found: {explicit}")
        return explicit
    try:
        import dockstring
    except ImportError:
        raise DockError(
            "could not import dockstring to locate the vendored Vina binary; "
            "install the 'docking' extra (uv sync --extra docking)."
        )
    bin_dir = os.path.join(os.path.dirname(dockstring.__file__), "resources", "bin")
    if sys.platform.startswith("linux"):
        candidates = [os.path.join(bin_dir, "vina_linux")]
    elif sys.platform == "darwin":
        candidates = [os.path.join(bin_dir, "vina_mac_catalina")]
    else:
        raise DockError(
            f"no vendored Vina binary is available for platform {sys.platform!r}; "
            "pass an explicit vina_bin path instead."
        )
    for c in candidates:
        if os.path.exists(c):
            return c
    raise DockError(f"no vendored Vina binary found under {os.path.dirname(dockstring.__file__)}")


def require(tool):
    if shutil.which(tool) is None:
        raise DockError(f"required tool '{tool}' not found on PATH.")


def convert(cmd, label):
    """Run an obabel conversion, raising DockError on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise DockError(
            f"{label} failed (obabel exit {res.returncode})\n"
            f"  cmd: {' '.join(cmd)}\n  stderr: {res.stderr.strip()}"
        )
    return res.stdout.strip()


def build_ligand_pdbqt(smiles, sdf_path, pdbqt_path):
    """SMILES -> 3D conformer (RDKit ETKDG + MMFF) -> SDF -> PDBQT (Open Babel)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DockError(f"could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xF1A9) != 0:
        raise DockError(f"RDKit 3D embedding failed for SMILES: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)  # fallback if MMFF params missing
    w = Chem.SDWriter(sdf_path)
    w.write(mol)
    w.close()
    convert(
        ["obabel", "-isdf", sdf_path, "-opdbqt", "-O", pdbqt_path],
        f"ligand SDF->PDBQT ({smiles})",
    )
    n = sum(1 for _ in open(pdbqt_path))
    if not any(l.startswith(("ROOT", "BRANCH")) for l in open(pdbqt_path)):
        print(
            f"vina_dock_core: WARNING -- ligand PDBQT has no torsion-tree records "
            f"({n} lines); Vina may treat it as rigid.",
            file=sys.stderr,
        )
    return n


def build_receptor_pdbqt(pdb_path, pdbqt_path):
    """receptor PDB -> rigid PDBQT (Open Babel).

    -h adds hydrogens (Vina only uses polar H, but extra H is harmless);
    -xr writes a RIGID molecule (no ROOT/BRANCH torsion tree). Without -xr,
    Open Babel writes the receptor as a flexible ligand with hundreds of
    BRANCH records, which Vina 1.1.2 rejects.
    """
    convert(
        ["obabel", "-ipdb", pdb_path, "-h", "-opdbqt", "-xr", "-O", pdbqt_path],
        "receptor PDB->PDBQT",
    )
    n = sum(1 for l in open(pdbqt_path) if l.startswith("ATOM"))
    if any(l.startswith(("ROOT", "BRANCH")) for l in open(pdbqt_path)):
        print(
            "vina_dock_core: WARNING -- receptor PDBQT still has torsion-tree "
            "records; Vina may fail to parse it.",
            file=sys.stderr,
        )
    return n


def run_vina(vina_bin, rec_pdbqt, lig_pdbqt, center, size, args, out_pdbqt, log_path):
    cmd = [
        vina_bin,
        "--receptor", rec_pdbqt,
        "--ligand", lig_pdbqt,
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
        "--out", out_pdbqt,
        "--log", log_path,
        "--exhaustiveness", str(args.exhaustiveness),
        "--num_modes", str(args.num_modes),
    ]
    if args.cpu:
        cmd += ["--cpu", str(args.cpu)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    print(f"vina_dock_core: running Vina...\n  {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise DockError(
            f"Vina failed (exit {res.returncode})\n"
            f"  stderr: {res.stderr.strip()}\n  stdout: {res.stdout.strip()}"
        )
    if res.stderr.strip():
        print(f"vina_dock_core: Vina stderr:\n{res.stderr.strip()}")
    return res.stdout


def parse_vina_log(log_path):
    """Return (best_affinity, n_modes) from a Vina log, or (None, 0) if unparseable."""
    if not os.path.exists(log_path):
        return None, 0
    row_re = re.compile(r"^\s*\d+\s+(-?\d+\.\d+)\s+")
    affinities = []
    with open(log_path) as fh:
        for line in fh:
            m = row_re.match(line)
            if m:
                affinities.append(float(m.group(1)))
    if not affinities:
        return None, 0
    return affinities[0], len(affinities)


def dock_at_centroid(
    receptor_pdb,
    smiles_list,
    center,
    box_size=22.0,
    exhaustiveness=8,
    num_modes=3,
    seed=0,
    cpu=0,
    overwrite_receptor=False,
    verbose=False,
):
    """Dock a list of ligands into a receptor at a KNOWN box center.

    Docks into a single box centered on `center` -- e.g. a cofolded
    ligand's own predicted centroid (see binding_energy.py), or any other
    known site.

    Persisted artefacts (next to the input PDB):
      - <stem>.pdbqt      rigid receptor (built once, reused across calls
                          unless overwrite_receptor=True)
      - <stem>_c<i>.sdf   top poses for molecule i

    Returns:
        A multi-line string report (header with receptor + center + box +
        receptor-PDBQT path, one block per molecule with score + pose-SDF
        path, and an overall best-molecule line). Failed molecules are
        marked and do not abort the rest.
    """
    print("vina_dock_core: docking at centroid...")
    print("==============================================")

    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    if not os.path.exists(receptor_pdb):
        raise DockError(f"receptor not found: {receptor_pdb}")
    require("obabel")
    vina_bin = find_vina_bin(None)  # always use dockstring's vendored copy
    receptor_pdb = os.path.abspath(receptor_pdb)
    stem = os.path.splitext(receptor_pdb)[0]
    rec_pdbqt = stem + ".pdbqt"
    center = [float(c) for c in center]
    box = [float(box_size)] * 3

    with contextlib.ExitStack() as stack:
        if not verbose:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            try:
                from rdkit import RDLogger

                RDLogger.DisableLog("rdApp.*")
                stack.callback(RDLogger.EnableLog, "rdApp.*")
            except Exception:
                pass

        if overwrite_receptor or not os.path.exists(rec_pdbqt):
            build_receptor_pdbqt(receptor_pdb, rec_pdbqt)

        work = tempfile.mkdtemp(prefix="dock_at_centroid_")
        vina_args = SimpleNamespace(
            exhaustiveness=exhaustiveness, num_modes=max(num_modes, 3), cpu=cpu, seed=seed
        )
        n_out_poses = min(3, vina_args.num_modes)
        results = []  # (idx, smi, affinity, sdf_path, status, detail)
        for idx, smi in enumerate(smiles_list):
            sdf_path = f"{stem}_c{idx}.sdf"
            try:
                lig_sdf = os.path.join(work, f"ligand_{idx}.sdf")
                lig_pdbqt = os.path.join(work, f"ligand_{idx}.pdbqt")
                build_ligand_pdbqt(smi, lig_sdf, lig_pdbqt)
                poses_pdbqt = os.path.join(work, f"poses_{idx}.pdbqt")
                log_path = os.path.join(work, f"vina_{idx}.log")
                run_vina(vina_bin, rec_pdbqt, lig_pdbqt, center, box, vina_args, poses_pdbqt, log_path)
                aff, nmodes = parse_vina_log(log_path)
                if aff is None:
                    results.append((idx, smi, None, sdf_path, "failed", "no score parsed from Vina log"))
                    continue
                convert(
                    ["obabel", "-ipdbqt", poses_pdbqt, "-osdf", "-O", sdf_path, "-l", str(n_out_poses)],
                    f"pose PDBQT->SDF (mol {idx}, top {n_out_poses})",
                )
                results.append((idx, smi, aff, sdf_path, "ok", f"{nmodes} modes"))
            except DockError as e:
                results.append((idx, smi, None, sdf_path, "failed", str(e)))
            except Exception as e:  # defensive: never let one molecule kill the run
                results.append((idx, smi, None, sdf_path, "error", f"{type(e).__name__}: {e}"))

    shutil.rmtree(work, ignore_errors=True)

    lines = []
    lines.append("Centroid docking report")
    lines.append(f"  receptor:         {receptor_pdb}")
    lines.append(f"  receptor PDBQT:   {rec_pdbqt}")
    lines.append(f"  vina binary:      {vina_bin}")
    lines.append(f"  box center:       ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    lines.append(f"  box size:         {box[0]:.1f} A (cubic)")
    lines.append("")
    lines.append(f"Molecules ({len(results)}):")
    for idx, smi, aff, sdf, status, detail in results:
        lines.append(f"  [{idx}] {smi}")
        if status == "ok":
            lines.append(f"        score: {aff:.2f} kcal/mol   ({detail})")
            lines.append(f"        poses SDF (top {n_out_poses}): {sdf}")
        else:
            lines.append(f"        {status}: {detail}")
    lines.append("")
    ok = [r for r in results if r[4] == "ok"]
    if ok:
        bidx, bsmi, baff, bsdf, _, _ = min(ok, key=lambda r: r[2])
        lines.append(f"Best molecule: [{bidx}] {bsmi}  score={baff:.2f} kcal/mol  SDF={bsdf}")
    else:
        lines.append("Best molecule: none docked successfully")
    return "\n".join(lines)
