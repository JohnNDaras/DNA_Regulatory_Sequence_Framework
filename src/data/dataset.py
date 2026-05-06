from bisect import bisect_left
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from .sequence_encoder import one_hot

class BedPeaksDataset(IterableDataset):
    def __init__(self, peaks_df, genome, context_length):
        super().__init__()
        self.context_length = context_length
        self.peaks_df = peaks_df
        self.genome = genome

    def __iter__(self):
        prev_end = 0
        prev_chrom = ""
        for row in self.peaks_df.itertuples():
            midpoint = int(0.5 * (row.start + row.end))
            seq = self.genome[row.chrom][midpoint - self.context_length // 2: midpoint + self.context_length // 2]
            if len(seq) == self.context_length:
                yield torch.tensor(one_hot(seq), dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)

            if prev_chrom == row.chrom and prev_end < row.start:
                midpoint = int(0.5 * (prev_end + row.start))
                seq = self.genome[row.chrom][midpoint - self.context_length // 2: midpoint + self.context_length // 2]
                if len(seq) == self.context_length:
                    yield torch.tensor(one_hot(seq), dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)

            prev_chrom = row.chrom
            prev_end = row.end

class BedPeaksDatasetBetter(IterableDataset):
    def __init__(self, atac_df, genome, context_length, n_neg=3, min_gap=2000, max_tries=20,
                 gc_tol=0.05, max_N_frac=0.1, rng_seed=1337, chroms_keep=None):
        super().__init__()
        self.context_length = context_length
        self.genome = genome
        self.n_neg = int(n_neg)
        self.min_gap = int(min_gap)
        self.max_tries = int(max_tries)
        self.gc_tol = float(gc_tol)
        self.max_N_frac = float(max_N_frac)
        self.rng = random.Random(rng_seed)

        if chroms_keep is not None:
            atac_df = atac_df[atac_df["chrom"].isin(chroms_keep)].copy()

        self.by_chrom = {}
        for chrom, dfc in atac_df.groupby("chrom"):
            starts = dfc["start"].values.astype(int)
            ends = dfc["end"].values.astype(int)
            idx = np.argsort(starts)
            self.by_chrom[chrom] = (starts[idx], ends[idx])
        self.chroms = [c for c in self.by_chrom.keys() if c in genome]

    def _ok_window(self, chrom, mid):
        L = self.context_length
        seq = self.genome[chrom][mid - L // 2: mid + L // 2]
        if len(seq) != L:
            return None
        if seq.count("N") / L > self.max_N_frac:
            return None
        return seq

    @staticmethod
    def _gc(seq):
        g = seq.count("G")
        c = seq.count("C")
        a = seq.count("A")
        t = seq.count("T")
        return (g + c) / max(a + c + g + t, 1)

    def _far_from_peaks(self, chrom, pos):
        starts, ends = self.by_chrom[chrom]
        i = bisect_left(starts, pos)
        for j in (i - 1, i):
            if 0 <= j < len(starts):
                peak_mid = (starts[j] + ends[j]) // 2
                if abs(pos - peak_mid) < self.min_gap:
                    return False
        return True

    def __iter__(self):
        for chrom in self.chroms:
            starts, ends = self.by_chrom[chrom]
            chr_len = len(self.genome[chrom])
            L = self.context_length

            for s, e in zip(starts, ends):
                mid = int((s + e) // 2)
                pos_seq = self._ok_window(chrom, mid)
                if pos_seq is None:
                    continue

                yield torch.tensor(one_hot(pos_seq), dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)

                target_gc = self._gc(pos_seq)
                got, tries = 0, 0
                while got < self.n_neg and tries < self.max_tries:
                    tries += 1
                    mid2 = self.rng.randint(L // 2, chr_len - L // 2 - 1)
                    if not self._far_from_peaks(chrom, mid2):
                        continue
                    neg_seq = self._ok_window(chrom, mid2)
                    if neg_seq is None:
                        continue
                    if abs(self._gc(neg_seq) - target_gc) > self.gc_tol:
                        continue
                    yield torch.tensor(one_hot(neg_seq), dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
                    got += 1
