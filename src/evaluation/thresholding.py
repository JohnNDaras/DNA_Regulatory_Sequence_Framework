import numpy as np
from sklearn.metrics import f1_score
def select_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        score = f1_score(y_true, (y_prob >= t).astype(int))
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t
