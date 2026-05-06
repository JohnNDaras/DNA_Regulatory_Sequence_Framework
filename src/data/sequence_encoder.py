import numpy as np
BASES = {"A": 0, "C": 1, "G": 2, "T": 3}
def one_hot(sequence: str) -> np.ndarray:
    arr = np.zeros((4, len(sequence)), dtype=np.float32)
    for i, ch in enumerate(sequence):
        idx = BASES.get(ch)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr
