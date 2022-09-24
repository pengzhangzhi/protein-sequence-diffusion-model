from torch.utils.data import Dataset, DataLoader
from .utils import load_seq_data
import torch


class SeqDataset(Dataset):
    def __init__(self, fas_dpath="./fas/", max_msa_depth=1000):
        self.seq_tns = load_seq_data(fas_dpath, max_msa_depth)
        self.max_seq_len = self.seq_tns.shape[-1]

    def __len__(self):
        return len(self.seq_tns)

    def __getitem__(self, idx):
        return self.seq_tns[idx]


def get_loader(args):
    dataset = SeqDataset(args.fas_dpath, args.max_msa_depth)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=3,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = add_data_arg(parser)
    args = parser.parse_args()
    loader = get_loader(args)
    for i, batch in enumerate(loader):
        print(batch.shape)
        break
