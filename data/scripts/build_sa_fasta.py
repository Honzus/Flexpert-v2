#!/usr/bin/env python3
"""Build a structure-aware (SA-pair) FASTA for Flexpert-3D from your own PDB files.

Each residue becomes an (uppercase AA, lowercase Foldseek 3Di) pair, so a chain of N residues is a
2N-character string, e.g. ``MdKvLp...``. This is the input format of ``train_3d.py`` and
``run_inference.py --backbone saprot`` (the bundled ``data/*_sa_sequences.fasta`` were made this way).

Requires the Foldseek binary (https://github.com/steineggerlab/foldseek). It is looked up from
``--foldseek``, then ``$FOLDSEEK``, then ``foldseek_bin`` in configs/data_config.yaml, then ``PATH``.

Run from the repository root:
    python3 -m data.scripts.build_sa_fasta --pdb_dir my_pdbs/ --out my_sa_sequences.fasta

Inputs: ``.pdb`` / ``.ent`` / ``.cif`` / ``.mmcif`` files (optionally ``.gz``); one chain per file
(the first, or ``--chain``). FASTA headers are the file names without extension
(``1abc_A.pdb`` -> ``>1abc_A``), or, with ``--names``, the names listed in that file (one per line,
e.g. ``1abc.A``; matched to ``1abc.A.pdb`` or ``1abc_A.pdb``) in the listed order — use that to make
headers match your splits JSON.
"""
import os
import sys
import glob
import gzip
import shutil
import argparse
import tempfile
import subprocess

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRUCTURE_EXTS = (".pdb", ".ent", ".cif", ".mmcif")


def structure_stem(path):
    """File name without the structure extension: ``1abc_A.cif.gz`` -> ``1abc_A``."""
    name = os.path.basename(path)
    if name.endswith(".gz"):
        name = name[:-3]
    for ext in STRUCTURE_EXTS:
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return os.path.splitext(name)[0]


# Get structural seqs from pdb file
def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = False,
                  plddt_threshold: float = 70.,
                  tmp_dir: str = None) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek
        path: Path to pdb file
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
        process_id: Process ID for temporary files. This is used for parallel processing.
        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.
        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.
        tmp_dir: Directory for Foldseek's temporary output (default: a fresh temp dir).

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs (in file order). The values are
        tuples of (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"Pdb file not found: {path}"

    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="build_sa_fasta_")
    tmp_save_path = os.path.join(tmp_dir, f"get_struc_seq_{process_id}.tsv")
    cmd = [foldseek, "structureto3didescriptor", "-v", "0", "--threads", "1",
           "--chain-name-mode", "1", path, tmp_save_path]
    subprocess.run(cmd, check=True)

    seq_dict = {}
    name = os.path.basename(path)
    plddts = extract_plddt(path) if plddt_mask else None
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            # Mask 3Di tokens of low-pLDDT residues with '#'
            if plddt_mask:
                plddt = plddts.get(chain)
                if plddt is None and i < len(plddts):  # e.g. blank chain IDs: match by order
                    plddt = list(plddts.values())[i]
                if plddt is None or len(plddt) != len(struc_seq):
                    raise ValueError(f"{path} chain {chain!r}: cannot match pLDDT values to the "
                                     f"{len(struc_seq)} Foldseek residues; run without --plddt_mask")
                struc_seq = "".join("#" if p < plddt_threshold else c
                                    for c, p in zip(struc_seq, plddt))

            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)

    if own_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        for p in (tmp_save_path, tmp_save_path + ".dbtype"):
            if os.path.exists(p):
                os.remove(p)
    return seq_dict


def extract_plddt(path: str) -> dict:
    """Per-residue pLDDT (mean atom B-factor) for each chain of the first model, in file order.

    Works for PDB and mmCIF (optionally gzipped) AlphaFold / ESMFold models. Only polymer residues
    with a CA atom are counted (as Foldseek does).
    """
    from Bio.PDB import MMCIFParser, PDBParser

    base = path[:-3] if path.endswith(".gz") else path
    parser = (MMCIFParser(QUIET=True) if base.lower().endswith((".cif", ".mmcif"))
              else PDBParser(QUIET=True))
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        model = next(iter(parser.get_structure("s", fh)))
    plddts = {}
    for chain in model:
        vals = [np.mean([a.get_bfactor() for a in res]) for res in chain
                if "CA" in res and (res.id[0] == " " or res.get_resname() == "MSE")]
        plddts[chain.id.strip()] = np.array(vals)
    return plddts


def resolve_foldseek(cli_value):
    """--foldseek, then $FOLDSEEK, then configs/data_config.yaml:foldseek_bin, then PATH."""
    candidates = [cli_value, os.environ.get("FOLDSEEK")]
    cfg_path = os.path.join(REPO_ROOT, "configs", "data_config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            candidates.append((yaml.safe_load(f) or {}).get("foldseek_bin"))
    candidates.append("foldseek")
    for c in candidates:
        if not c:
            continue
        found = c if os.path.isfile(c) and os.access(c, os.X_OK) else shutil.which(c)
        if found:
            return os.path.abspath(found)
    return None


def is_bundled_fasta(out_path):
    """True for the shipped data/<dataset>_sa_sequences.fasta files, which must never be overwritten."""
    out_abs = os.path.realpath(out_path)
    data_dir = os.path.realpath(os.path.join(REPO_ROOT, "data"))
    return (os.path.dirname(out_abs) == data_dir
            and os.path.basename(out_abs).endswith("_sa_sequences.fasta"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb_dir", required=True, help="Directory with the input structures.")
    ap.add_argument("--out", required=True, help="Output SA-pair FASTA.")
    ap.add_argument("--force", action="store_true", help="Overwrite --out if it exists "
                    "(the bundled data/*_sa_sequences.fasta are never overwritten).")
    ap.add_argument("--foldseek", default=None,
                    help="Foldseek binary (default: $FOLDSEEK, configs/data_config.yaml foldseek_bin, PATH).")
    ap.add_argument("--chain", default=None,
                    help="Chain ID to extract from every file (default: the first chain in each file).")
    ap.add_argument("--names", default=None,
                    help="Optional file with one name per line: restricts and orders the output, "
                         "and is used for the FASTA headers.")
    ap.add_argument("--glob", default=None,
                    help="File pattern inside --pdb_dir (default: all .pdb/.ent/.cif/.mmcif[.gz] files).")
    ap.add_argument("--plddt_mask", action="store_true",
                    help="Mask 3Di tokens of low-pLDDT residues (AlphaFold/ESMFold models) with '#'.")
    ap.add_argument("--plddt_threshold", type=float, default=70.0)
    return ap.parse_args()


def main():
    args = parse_args()

    if is_bundled_fasta(args.out):
        sys.exit(f"Refusing to write {args.out}: it is a bundled dataset FASTA. Choose another --out.")
    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} exists; pass --force to overwrite it or choose another --out.")

    foldseek = resolve_foldseek(args.foldseek)
    if foldseek is None:
        sys.exit("Foldseek binary not found. Install it (conda install -c conda-forge -c bioconda "
                 "foldseek, or the static binary from https://mmseqs.com/foldseek/) and pass "
                 "--foldseek /path/to/foldseek, set $FOLDSEEK, or put it on PATH.")

    if args.glob:
        pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, args.glob)))
    else:
        pdb_files = sorted(p for p in glob.glob(os.path.join(args.pdb_dir, "*"))
                           if p.lower().removesuffix(".gz").endswith(STRUCTURE_EXTS))
    by_stem = {}
    for p in pdb_files:
        stem = structure_stem(p)
        if stem in by_stem:
            sys.exit(f"Two structures share the name {stem!r}: {by_stem[stem]} and {p}. "
                     "Rename one or use --glob.")
        by_stem[stem] = p
    if args.names:
        with open(args.names) as f:
            names = [line.strip() for line in f if line.strip()]
        jobs, missing = [], []
        for name in names:
            path = by_stem.get(name) or by_stem.get(name.replace(".", "_"))
            if path is None:
                missing.append(name)
            else:
                jobs.append((name, path))
        if missing:
            print(f"Warning: no structure in {args.pdb_dir} for {len(missing)} name(s), "
                  f"e.g. {missing[:5]}", file=sys.stderr)
    else:
        jobs = list(by_stem.items())
    if not jobs:
        sys.exit(f"No structures matching {args.glob or '*.pdb/*.cif'!r} found in {args.pdb_dir}.")

    tmp_dir = tempfile.mkdtemp(prefix="build_sa_fasta_")
    written, failed = 0, []
    try:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as out:
            for name, path in jobs:
                chains = [args.chain] if args.chain else None
                seq_dict = get_struc_seq(foldseek, path, chains=chains, plddt_mask=args.plddt_mask,
                                         plddt_threshold=args.plddt_threshold, tmp_dir=tmp_dir)
                if not seq_dict:
                    failed.append(name)
                    continue
                combined = next(iter(seq_dict.values()))[-1]
                out.write(f">{name}\n{combined}\n")
                written += 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if failed:
        print(f"Warning: no matching chain for {len(failed)} structure(s), e.g. {failed[:5]}",
              file=sys.stderr)
    print(f"Wrote {written} SA-pair sequences to {args.out}")


if __name__ == "__main__":
    main()
