from torch.utils.data import DataLoader
from .dataset import BedPeaksDataset
def build_dataloaders(train_df, val_df, genome, context_length, batch_size=512, num_workers=0):
    return (
        DataLoader(BedPeaksDataset(train_df, genome, context_length), batch_size=batch_size, num_workers=num_workers),
        DataLoader(BedPeaksDataset(val_df, genome, context_length), batch_size=batch_size, num_workers=num_workers),
    )
