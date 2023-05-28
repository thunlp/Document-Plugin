import json
import os
from torch.utils.data import Dataset

class JsonDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data = [json.loads(line) for line in open(config.get("data", "%s_data_path" % mode), "r", encoding="utf8")]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
