import json
import os
from tqdm import tqdm
import kara_storage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', '-in', default="data/c4-json", required=True)
parser.add_argument('--out_path', '-out', default="data/c4-kara", required=True)
args = parser.parse_args()



out_path = args.out_path
if os.path.exists(out_path):
    os.system("rm -rf %s" % out_path)
os.makedirs(out_path, exist_ok=True)

storage = kara_storage.KaraStorage("file://%s" % out_path)
dataset = storage.open("C4", "train", "w", version="1st")

valid_text = []
c4_path = args.in_path
for fname in tqdm(os.listdir(c4_path)):
    fin = open(os.path.join(c4_path, fname), "r")
    try:
        lines = fin.readlines()
    except Exception as err:
        print(err)
        print(fname)
        continue
    for line in tqdm(lines):
        line = json.loads(line)
        if len(line["text"].split()) < 50:
            continue
        dataset.write(line)
dataset.close()

