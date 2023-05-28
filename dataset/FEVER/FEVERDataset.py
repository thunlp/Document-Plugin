import json
import os
from torch.utils.data import Dataset
from tools import print_rank

class FEVERDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data = []
        for line in open(config.get("data", "%s_data_path" % mode), "r", encoding="utf8"):
            line = json.loads(line)
            line["output"][0]["provenance"] = line["output"][0]["provenance"][:3]
            self.data.append(line)
        print_rank("Load %s data from %s, size: %d" % (mode, config.get("data", "%s_data_path" % mode), len(self.data)))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
