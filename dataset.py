import pandas as pd
from torch.utils.data import Dataset

class CFGDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "smiles": row["smiles"],
            "qed": row["qed"],
            "logp": row["logp"],
            "tpsa": row["tpsa"],
            "sas": row["sas"],
        }
