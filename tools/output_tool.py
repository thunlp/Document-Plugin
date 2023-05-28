from genericpath import exists
import json
import os


def null_output_function(data, config, *args, **params):
    return str(data)


def binary_output_function(data, config, *args, **params):
    if data['total'] == 0:
        metric = {'acc': 0}
    else:
        metric = {'acc': round(data['right'] / data['total'], 4)}
    return json.dumps(metric)



def squad_output_function(data, config, *args, **params):
    if data["train"]:
        acc = round(data["right"] / data["total"], 4)
        return json.dumps({"tok_acc": acc})
    else:
        if data['NA_tp'] != 0 or data['NA_fp'] != 0:
            pre = float(data['NA_tp']) / (data['NA_tp'] + data["NA_fp"])
            recall = float(data['NA_tp']) / (data['NA_tp'] + data["NA_fn"])
            if pre + recall == 0:
                naf1 = 0
            else:
                naf1 = 2 * pre * recall / (pre + recall)
        else:
            naf1 = 0

        return json.dumps({
            "EM": round(data["em_sum"] / data["total"], 4),
            "F1": round(data["f1_sum"] / data["total"], 4),
            "NA_F1": round(naf1, 4),
            "ROUGE-L-F": round(data["ROUGE-L-F"] / data["total"], 4) if "ROUGE-L-F" in data else 0,
            "ROUGE-L-P": round(data["ROUGE-L-P"] / data["total"], 4) if "ROUGE-L-P" in data else 0,
            "ROUGE-L-R": round(data["ROUGE-L-R"] / data["total"], 4) if "ROUGE-L-R" in data else 0,
            }
        )

def mlm_output_function(data, config, *args, **params):
    acc = round(data["right"] / data["total"], 4)
    return json.dumps({"tok_acc": acc, "avg_loss": sum(data["loss"]) / len(data["loss"])})
