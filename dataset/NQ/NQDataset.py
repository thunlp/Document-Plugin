import json
import os
from torch.utils.data import Dataset

class NQDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = [json.loads(line) for line in open(config.get("data", "%s_data_path" % mode), "r", encoding="utf8")]

    def __getitem__(self, idx):
        ret = self.data[idx]
        return {
            "context": ret["passage"],
            "answers": [{"text": a} for a in ret["answers"]],
            "question": ret["question"],
        }

    def __len__(self):
        return len(self.data)
