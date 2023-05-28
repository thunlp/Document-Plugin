import json
import os
from torch.utils.data import Dataset

class JsonlineDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        data_path = config.get("data", "%s_data_path" % mode)
        fin = open(data_path, "r")
        self.data = [json.loads(line) for line in fin]
        print("the number of data in %s: %s" % (mode, len(self.data)))

    def __getitem__(self, idx):
        return self.data[idx]
        
    def __len__(self):
        return len(self.data)
