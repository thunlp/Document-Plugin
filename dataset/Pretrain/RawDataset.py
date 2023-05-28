import json
import os
from torch.utils.data import Dataset

class RawDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        data = json.load(open(config.get("data", "%s_data_path" % mode), "r", encoding="utf8"))
        self.data = [{"doc": d["document"], "id": did, "question": d["questions"]} for did, d in enumerate(data) if len(d["questions"]) > 0]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
