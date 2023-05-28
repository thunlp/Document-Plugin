import json
import os
import random
from tqdm import tqdm

class KGDataset():
    def __init__(self):
        fin = open("/liuzyai04/thunlp/xcj/docaspara/data/wikidata5m/wikidata5m_transductive_train.txt", "r")
        self.triples = []
        self.entity2ids = {}

        allents = set(json.load(open("/liuzyai04/thunlp/xcj/docaspara/data/knowledge-task/knowledge-task/Wiki80/allent.json", "r")))
        self.rel2id = json.load(open("/liuzyai04/thunlp/xcj/docaspara/data/knowledge-task/knowledge-task/Wiki80/pid2name.json", "r"))
        word2qid = json.load(open("/liuzyai04/thunlp/xcj/docaspara/data/wikidata5m/word2qid.json", "r"))
        self.qid2word = {word2qid[name]: name.replace("_", " ") for name in word2qid}

        for line in tqdm(fin.readlines()):
            line = line.strip().split("\t")
            if line[1] not in self.rel2id:
                continue
            if line[0] not in self.qid2word or line[2] not in self.qid2word:
                continue
            self.triples.append(line)

            # if line[0] in allents:
            if line[0] not in self.entity2ids:
                self.entity2ids[line[0]] = []
            self.entity2ids[line[0]].append(len(self.triples) - 1)
            # if line[2] in allents:
            if line[2] not in self.entity2ids:
                self.entity2ids[line[2]] = []
            self.entity2ids[line[2]].append(len(self.triples) - 1)
    
    # def search_path_recursive(self, qid, tid, nowid=None, nowpath=[]):
    #     if len(nowpath) == 3:
    #         return []
    #     depth = len(nowpath)
    #     if nowid is None and depth == 0:
    #         nowid = qid
    #     nextup = [self.triples[i] for i in self.entity2ids[nowid]]
    #     nextent = set([tup[0] for tup in nextup] + [tup[2] for tup in nextup]) - set([nowid])
    #     if tid in nextent:

    
    def search_path(self, qid, tid):
        if qid not in self.entity2ids or tid not in self.entity2ids:
            return ""
        headtup = [self.triples[i] for i in self.entity2ids[qid]]
        tailtup = [self.triples[i] for i in self.entity2ids[tid]]
        head_relate_ent = set([tup[0] for tup in headtup] + [tup[2] for tup in headtup]) - set([qid, tid])
        tail_relate_ent = set([tup[0] for tup in tailtup] + [tup[2] for tup in tailtup]) - set([qid, tid])
        hop2end = head_relate_ent & tail_relate_ent
        if len(hop2end) == 0:
            return ""
        relate_tup = [self.triples[i] for i in self.entity2ids[qid] if self.triples[i][0] in hop2end or self.triples[i][2] in hop2end] + \
                    [self.triples[i] for i in self.entity2ids[tid] if self.triples[i][0] in hop2end or self.triples[i][2] in hop2end]
        sents = []
        for tup in relate_tup:
            if tup[0] != tid and tup[2] != tid:
                sents.append("%s is %s of %s." % (self.qid2word[tup[0]], self.rel2id[tup[1]][0], self.qid2word[tup[2]]))
        return " ".join(sents)
    
    def get_relations(self, qid, tid=None):
        if not qid in self.entity2ids:
            return ""
        rels = self.entity2ids[qid][:10]
        sents = []
        for tupid in rels:
            tup = self.triples[tupid]
            if tup[0] != tid and tup[2] != tid:
                sents.append("%s is %s of %s." % (self.qid2word[tup[0]], self.rel2id[tup[1]][0], self.qid2word[tup[2]]))
        return " ".join(sents)

