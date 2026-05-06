import pickle
def load_genome(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
