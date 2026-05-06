import pandas as pd
def build_benchmark_table(records):
    return pd.DataFrame(records).sort_values("val_auprc", ascending=False)
