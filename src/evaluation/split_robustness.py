def alternate_ctcf_split(df):
    train = df[df["chrom"].isin(["chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"])]
    val = df[df["chrom"].isin(["chr3","chr4"])]
    test = df[df["chrom"].isin(["chr1","chr2"])]
    return train, val, test
