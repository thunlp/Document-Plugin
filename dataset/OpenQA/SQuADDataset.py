import json
import os
from torch.utils.data import Dataset
import random

class SQuADDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        data = json.load(open(config.get("data", "%s_data_path" % mode), "r", encoding="utf8"))
        self.qas = []
        self.context = []
        for doc in data["data"]:
            title = doc["title"]
            for para in doc["paragraphs"]:
                context = para["context"]
                self.context.append({"text": context})
                qas = []
                for qa in para["qas"]:
                    qa.update({"context": len(self.context) - 1})
                    if "is_impossible" in qa and qa["is_impossible"]:
                        qa["answers"] = [{"text": "no answer"}]
                    qa["answers"] = [a["text"] for a in qa["answers"]]
                    qa["title"] = title
                    qas.append(qa)
                self.qas.extend(qas)

    def __getitem__(self, idx):
        qa = self.qas[idx]
        ret = qa.copy()
        ret["context"] = [self.context[qa["context"]]]
        # print(ret)
        return ret

    def __len__(self):
        # return 320
        return len(self.qas)


class FewSQuADDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.ratio = config.getfloat("train", "few_ratio")
        data = json.load(open(config.get("data", "%s_data_path" % mode), "r", encoding="utf8"))
        self.qas = []
        self.context = []
        if mode == "train":
            random.seed(10086)
            doces = random.sample(data["data"], int(self.ratio * len(data["data"])))
        else:
            doces = data["data"]
        # for doc in data["data"]:
        for doc in doces:
            title = doc["title"]
            for para in doc["paragraphs"]:
                context = para["context"]
                self.context.append({"text": context})
                qas = []
                for qa in para["qas"]:
                    qa.update({"context": len(self.context) - 1})
                    if "is_impossible" in qa and qa["is_impossible"]:
                        qa["answers"] = [{"text": "no answer"}]
                    qa["answers"] = [a["text"] for a in qa["answers"]]
                    qa["title"] = title
                    qas.append(qa)
                self.qas.extend(qas)

    def __getitem__(self, idx):
        qa = self.qas[idx % len(self.qas)]
        ret = qa.copy()
        ret["context"] = [self.context[qa["context"]]]
        # print(ret)
        return ret

    def __len__(self):
        # return 320
        if self.mode == "train":
            return int(len(self.qas) * 0.1 / self.ratio)
        else:
            return len(self.qas)

