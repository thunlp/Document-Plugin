import json
import os
from torch.utils.data import Dataset

class OpenQADataset2(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        
        self.data = []
        fin = open(config.get("data", "%s_data_path" % mode), "r")
        fin.readline()
        for line in fin:
            line = json.loads(line)
            for qa in line["qas"]:
                self.data.append({
                    "context": [{"text": line["context"]}],
                    "question": qa["question"],
                    "answers": qa["answers"]
                })

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
