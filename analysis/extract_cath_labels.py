"""Extract a per-PDB CATH classification (Class + Architecture) for shading the universes.

Reads the local CATH domain list (``data/mdCATH/mdCATH/cath-domain-list.txt``), where each row is
``<domain_id> C A T H S O L I D length resolution`` and ``domain_id`` is ``<pdbid><chain><dom>``
(e.g. ``1oaiA00``). A PDB can hold several domains spanning different folds, so we collapse to one
label per PDB ID by **residue-weighted dominant node**: sum domain length per node and take argmax,
for both Class (``C``) and Architecture (``C.A``).

Architecture names come from ``cath-names.txt`` (downloaded once if absent; falls back to the
numeric ``C.A`` code when offline). Class names are the four standard CATH classes.

Output: ``data/pdb_cath_labels.tsv`` with columns
``pdbid  class  class_name  arch  arch_name  n_domains  n_classes``.
"""
import argparse
import os
from collections import defaultdict

CLASS_NAMES = {
    1: "Mainly Alpha",
    2: "Mainly Beta",
    3: "Alpha-Beta",
    4: "Few Secondary Structures",
    6: "Special",
}

CATH_NAMES_URL = (
    "http://download.cathdb.info/cath/releases/latest-release/"
    "cath-classification-data/cath-names.txt"
)


def load_arch_names(cath_names_path, download=True):
    """Return {"C.A": name} for architecture-level CATH nodes.

    cath-names.txt rows look like: ``1.10  1oaiA00  :Orthogonal Bundle`` — node id, a
    representative domain, then ``:`` + the human-readable name. We keep nodes whose id has
    exactly two dotted fields (i.e. ``C.A``).
    """
    if not os.path.exists(cath_names_path) and download:
        try:
            import requests

            print(f"Downloading cath-names.txt -> {cath_names_path}")
            r = requests.get(CATH_NAMES_URL, timeout=60)
            r.raise_for_status()
            with open(cath_names_path, "w") as f:
                f.write(r.text)
        except Exception as e:  # offline-safe: fall back to numeric arch codes
            print(f"  cath-names download failed ({e}); using numeric architecture codes.")
            return {}

    if not os.path.exists(cath_names_path):
        return {}

    arch = {}
    with open(cath_names_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split(None, 2)
            if len(parts) < 3:
                continue
            node = parts[0]
            if node.count(".") != 1:  # keep only "C.A"
                continue
            name = parts[2].lstrip(":").strip()
            arch[node] = name
    print(f"  loaded {len(arch)} architecture names")
    return arch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cath_list", default="data/mdCATH/mdCATH/cath-domain-list.txt")
    ap.add_argument("--cath_names", default="data/cath-names.txt")
    ap.add_argument("--output", default="data/pdb_cath_labels.tsv")
    ap.add_argument("--no_download", action="store_true",
                    help="Do not attempt to fetch cath-names.txt (use numeric arch codes).")
    args = ap.parse_args()

    arch_names = load_arch_names(args.cath_names, download=not args.no_download)

    # residue totals per node, per PDB
    res_class = defaultdict(lambda: defaultdict(int))  # pdbid -> {C: residues}
    res_arch = defaultdict(lambda: defaultdict(int))   # pdbid -> {"C.A": residues}
    ndom = defaultdict(int)
    n_rows = 0
    with open(args.cath_list) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            dom = p[0]
            C, A, length = int(p[1]), int(p[2]), int(p[10])
            pdbid = dom[:4].lower()
            res_class[pdbid][C] += length
            res_arch[pdbid][f"{C}.{A}"] += length
            ndom[pdbid] += 1
            n_rows += 1
    print(f"Parsed {n_rows:,} domains across {len(res_class):,} PDB ids")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as out:
        out.write("pdbid\tclass\tclass_name\tarch\tarch_name\tn_domains\tn_classes\n")
        for pdbid in sorted(res_class):
            cls = max(res_class[pdbid].items(), key=lambda kv: kv[1])[0]
            arch = max(res_arch[pdbid].items(), key=lambda kv: kv[1])[0]
            out.write(
                f"{pdbid}\t{cls}\t{CLASS_NAMES.get(cls, str(cls))}\t"
                f"{arch}\t{arch_names.get(arch, arch)}\t"
                f"{ndom[pdbid]}\t{len(res_class[pdbid])}\n"
            )
    print(f"Wrote {len(res_class):,} PDB labels -> {args.output}")


if __name__ == "__main__":
    main()
