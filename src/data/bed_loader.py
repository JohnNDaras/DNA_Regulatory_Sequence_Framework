import pandas as pd
def load_bed(path: str, include_chr_x_y: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer", usecols=range(3), names=["chrom", "start", "end"])
    if not include_chr_x_y:
        df = df[~df["chrom"].isin(["chrX", "chrY"])]
    return df.sort_values(["chrom", "start"]).drop_duplicates().reset_index(drop=True)
