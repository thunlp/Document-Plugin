import json
import os
from torch.utils.data import Dataset

class OpenQADataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data = []
        self.ctxnum = 3
        fin = open(config.get("data", "%s_data_path" % mode), "r")
        for line in fin:
            line = json.loads(line)
            question = line["input"]
            ctxs = line["output"][0]["provenance"][:self.ctxnum]
            answer = [l["answer"] for l in line["output"][1:]]
            self.data.append({
                "context": ctxs,
                "question": question,
                "answers": answer
            })

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


import random
class FewOpenQADataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        
        data = []
        self.ctxnum = 3
        fin = open(config.get("data", "%s_data_path" % mode), "r")
        for line in fin:
            line = json.loads(line)
            question = line["input"]
            ctxs = line["output"][0]["provenance"][:self.ctxnum]
            answer = [l["answer"] for l in line["output"][1:]]
            data.append({
                "context": ctxs,
                "question": question,
                "answers": answer
            })

        self.few_num = config.getint("fewshot", "few_num")
        self.seed = config.getint("fewshot", "dataset_seed")
        if mode == "train":
            random.seed(self.seed)
            self.data = random.sample(data, self.few_num)
        else:
            self.data = data

    def __getitem__(self, idx):
        return self.data[idx % len(self.data)]

    def __len__(self):
        return max(200, len(self.data))
