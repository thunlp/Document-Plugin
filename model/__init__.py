from .MLM.PlugDPL import PlugDPlugLearning
from .TextClassification.TextClassifier import TextClassifier
from .Seq2Seq.Seq2Seq import Seq2Seq


model_list = {
    "TextClassification": TextClassifier,
    "Seq2Seq": Seq2Seq,
    "PlugD": PlugDPlugLearning,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
