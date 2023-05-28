import json
import os
from torch.utils.data import Dataset

class SFDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        
        self.data = []
        self.ctxnum = 3
        fin = open(config.get("data", "%s_data_path" % mode), "r")
        self.model_type = config.get("model", "model_type")
        for line in fin:
            line = json.loads(line)
            question = line["input"]
            ctxs = line["output"][0]["provenance"][:self.ctxnum]
            if mode == "train" and "Post" in self.model_type:
                answer = [line["output"][1]["answer"]]
            else:
                answer = [l["answer"] for l in line["output"][1:]]
            ent0, ent1 = question.split("[SEP]")
            self.data.append({
                "context": ctxs,
                "question": f"the {ent1} of {ent0} is ", # question.replace("[SEP]", " * "),
                "answers": answer
            })

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
