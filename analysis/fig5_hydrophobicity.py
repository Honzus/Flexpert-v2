"""Figure 5d-f: residue hydrophobicity (Kyte-Doolittle) vs predicted flexibility.

For every amino acid the mean Flexpert prediction over all its residues is plotted against its
Kyte-Doolittle hydrophobicity; Spearman's rho is computed over these 20 per-amino-acid means.
For the PDB set, runs of >= 4 consecutive His (expression tags) are masked; the mdCATH domains
are not masked (as in the paper; ``--his_tag_min_run`` overrides).

  --set pdb    : all residues of the annotated PDB chains (Fig. 5d).
  --set mdcath : mdCATH domains split by relative solvent accessibility, buried (rSASA <= 0.25,
                 Fig. 5e) and exposed (rSASA > 0.25, Fig. 5f). rSASA = Shrake-Rupley SASA
                 (biotite, ``vdw_radii="Single"``, all atoms of the mdCATH PDB) summed per residue,
                 divided by the Tien et al. (2013) theoretical maximum.

    python3 -m analysis.fig5_hydrophobicity --set pdb
    python3 -m analysis.fig5_hydrophobicity --set mdcath
    # -> analysis/figures/fig5d_hydrophobicity_pdb.png, fig5{e,f}_hydrophobicity_{buried,exposed}.png,
    #    per_aa_<set>.csv
"""
import argparse
import os
import re

import numpy as np

from analysis.plot_flexibility_universe import read_flexpert_predictions

KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
# Tien et al. 2013, theoretical maximum ASA (Å^2).
MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0, 'GLN': 225.0,
    'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
    'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0, 'THR': 172.0, 'TRP': 285.0,
    'TYR': 263.0, 'VAL': 174.0,
}
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
CHARMM_HIS = {'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS'}
GROUPS = [  # legend label, residues, colour
    ('Special (G, P)', 'GP', 'tab:orange'),
    ('Charged (D, E, K, R, H)', 'DEKRH', 'tab:red'),
    ('Aromatic (F, W, Y)', 'FWY', 'tab:purple'),
    ('Aliphatic hydrophobic (A, V, L, I, M, C)', 'AVLIMC', 'tab:blue'),
    ('Polar (S, T, N, Q)', 'STNQ', 'tab:green'),
]


def his_tag_mask(seq, min_run=4):
    """True for residues inside a run of >= ``min_run`` consecutive His (0 disables)."""
    mask = np.zeros(len(seq), dtype=bool)
    if min_run:
        for m in re.finditer(f'H{{{min_run},}}', seq):
            mask[m.start():m.end()] = True
    return mask


def per_aa_stats(seqs, values):
    """``{aa: (mean, sem, n)}`` over all residues; ``seqs``/``values`` are parallel lists."""
    aa = np.frombuffer(''.join(seqs).encode(), dtype='S1').astype(str)
    v = np.concatenate(values).astype(np.float64)
    out = {}
    for a in KYTE_DOOLITTLE:
        x = v[(aa == a) & np.isfinite(v)]
        out[a] = (x.mean(), x.std(ddof=1) / np.sqrt(x.size), x.size)
    return out


def residue_rsasa(pdb_path):
    """``(sequence, rSASA per residue)`` for an mdCATH PDB (all atoms, CHARMM His names)."""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    atoms = PDBFile.read(pdb_path).get_structure(model=1)
    atoms.res_name = np.array([CHARMM_HIS.get(r, r) for r in atoms.res_name])
    sasa = struc.apply_residue_wise(atoms, struc.sasa(atoms, vdw_radii='Single'), np.nansum)
    _, names = struc.get_residues(atoms)
    seq = ''.join(THREE_TO_ONE.get(r, 'X') for r in names)
    rsasa = np.array([s / MAX_ASA.get(r, np.nan) for s, r in zip(sasa, names)])
    return seq, rsasa


def _rsasa_job(args):
    name, pdb_dir = args
    try:
        return name, *residue_rsasa(os.path.join(pdb_dir, f'{name}.pdb'))
    except Exception as e:  # unreadable structure: reported and skipped
        return name, None, str(e)


def load_rsasa(names, pdb_dir, cache, workers):
    """``{name: (sequence, rsasa)}``, computed in parallel and cached as an .npz."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        done = dict(zip(z['names'], zip(z['seqs'], z['rsasa'])))
        if set(names) <= set(done):
            return {n: done[n] for n in names}
    from multiprocessing import Pool

    from tqdm import tqdm

    out = {}
    with Pool(workers) as pool:
        for name, seq, r in tqdm(pool.imap_unordered(_rsasa_job, [(n, pdb_dir) for n in names],
                                                     chunksize=8), total=len(names)):
            if seq is None:
                print(f'  skip {name}: {r}')
            else:
                out[name] = (seq, r)
    keys = list(out)
    np.savez(cache, names=np.array(keys), seqs=np.array([out[k][0] for k in keys]),
             rsasa=np.array([out[k][1] for k in keys], dtype=object))
    return out


def plot_scatter(stats, title, out_path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.stats import spearmanr

    aas = list(KYTE_DOOLITTLE)
    kd = np.array([KYTE_DOOLITTLE[a] for a in aas])
    mean = np.array([stats[a][0] for a in aas])
    rho = spearmanr(kd, mean)[0]
    colour = {a: c for _, res, c in GROUPS for a in res}

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    slope, icpt = np.polyfit(kd, mean, 1)
    xs = np.array([kd.min(), kd.max()])
    ax.plot(xs, slope * xs + icpt, ls='--', color='dimgray', lw=1.3, zorder=1)
    for a, x, y in zip(aas, kd, mean):
        ax.scatter(x, y, s=260, color=colour[a], edgecolor='black', lw=0.6, zorder=2)
        ax.text(x, y, a, ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=3)
    ax.set_xlabel('Kyte-Doolittle hydrophobicity')
    ax.set_ylabel('mean Flexpert prediction (Å)')
    ax.set_title(f'{title}\n(Spearman ρ = {rho:.2f})')
    ax.grid(alpha=0.3)
    ax.legend(handles=[Patch(color=c, label=l) for l, _, c in GROUPS], fontsize=8,
              loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'  {title}: Spearman rho = {rho:.3f} (20 amino-acid means), '
          f'n = {sum(s[2] for s in stats.values()):,} residues')
    print(f'  wrote {out_path}')
    return rho


def write_csv(path, columns):
    """``columns``: ``{prefix: {aa: (mean, sem, n)}}``."""
    with open(path, 'w') as f:
        f.write('aa,kyte_doolittle,' + ','.join(f'{p}_mean,{p}_sem,{p}_n' for p in columns) + '\n')
        for a in KYTE_DOOLITTLE:
            f.write(f'{a},{KYTE_DOOLITTLE[a]},' + ','.join(
                f'{s[a][0]:.6f},{s[a][1]:.6f},{s[a][2]}' for s in columns.values()) + '\n')
    print(f'  wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', required=True, choices=['pdb', 'mdcath'])
    ap.add_argument('--predictions', default=None,
                    help='Default: outputs/predictions/flexpert_seq_{pdb_sequences,mdCATH_sequences}.txt')
    ap.add_argument('--fasta_path', default='data/pdb_sequences.fasta', help='--set pdb only.')
    ap.add_argument('--pdb_dir', default='data/mdCATH/mdCATH/pdbs', help='--set mdcath only.')
    ap.add_argument('--rsasa_threshold', type=float, default=0.25)
    ap.add_argument('--his_tag_min_run', type=int, default=None,
                    help='Mask His runs of at least this length; 0 disables. Default: 4 for pdb, 0 for mdcath.')
    ap.add_argument('--workers', type=int, default=16, help='Processes for the rSASA computation.')
    ap.add_argument('--fig_dir', default='analysis/figures')
    ap.add_argument('--limit', type=int, default=None, help='Only the first N chains (smoke test).')
    args = ap.parse_args()

    stem = 'pdb_sequences' if args.set == 'pdb' else 'mdCATH_sequences'
    if args.his_tag_min_run is None:
        args.his_tag_min_run = 4 if args.set == 'pdb' else 0
    preds = read_flexpert_predictions(args.predictions or f'outputs/predictions/flexpert_seq_{stem}.txt',
                                      limit=args.limit)
    os.makedirs(args.fig_dir, exist_ok=True)

    if args.set == 'pdb':
        from run_inference import clean_aa, read_fasta

        fasta = {n.replace('.', '_'): clean_aa(s) for n, s in read_fasta(args.fasta_path).items()}
        seqs, vals, skipped = [], [], 0
        for name, v in preds.items():
            s = fasta.get(name)
            if s is None or len(s) != len(v):
                skipped += 1
                continue
            v = np.where(his_tag_mask(s, args.his_tag_min_run), np.nan, v)
            seqs.append(s)
            vals.append(v)
        print(f'{len(seqs):,} chains used ({skipped} skipped: missing sequence or length mismatch)')
        stats = per_aa_stats(seqs, vals)
        write_csv(os.path.join(args.fig_dir, 'per_aa_pdb.csv'), {'all': stats})
        plot_scatter(stats, 'Annotated PDB chains, all residues',
                     os.path.join(args.fig_dir, 'fig5d_hydrophobicity_pdb.png'))
        return

    cache = os.path.join(args.fig_dir, 'mdcath_rsasa.npz')
    rsasa = load_rsasa(list(preds), args.pdb_dir, cache, args.workers)
    split = {'buried': ([], []), 'exposed': ([], [])}
    skipped = 0
    for name, v in preds.items():
        if name not in rsasa or len(rsasa[name][0]) != len(v):
            skipped += 1
            continue
        s, r = rsasa[name]
        v = np.where(his_tag_mask(s, args.his_tag_min_run), np.nan, v)
        for key, sel in (('buried', r <= args.rsasa_threshold), ('exposed', r > args.rsasa_threshold)):
            split[key][0].append(s)
            split[key][1].append(np.where(sel, v, np.nan))
    print(f'{len(preds) - skipped:,} domains used ({skipped} skipped: no structure or length mismatch)')
    stats = {k: per_aa_stats(*split[k]) for k in split}
    write_csv(os.path.join(args.fig_dir, 'per_aa_mdcath.csv'), stats)
    pct = f'{args.rsasa_threshold:.0%}'
    plot_scatter(stats['buried'], f'mdCATH buried residues (rSASA ≤ {pct})',
                 os.path.join(args.fig_dir, 'fig5e_hydrophobicity_buried.png'))
    plot_scatter(stats['exposed'], f'mdCATH exposed residues (rSASA > {pct})',
                 os.path.join(args.fig_dir, 'fig5f_hydrophobicity_exposed.png'))


if __name__ == '__main__':
    main()
