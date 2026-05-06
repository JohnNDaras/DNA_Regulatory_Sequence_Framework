class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False
    def step(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
