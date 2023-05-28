import json
from re import I
import torch
import os
import numpy as np

import random
from transformers import T5Tokenizer, T5Config
from transformers.file_utils import is_torch_fx_proxy
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize

from bmtrain import print_rank
import logging
from tools import output_log, shift_tokens_right
from collections import namedtuple, defaultdict
logger = logging.getLogger(__name__)

STOPWORDS = {'', "the", ",", ";", ".", "(",
             'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'also', 'could', 'would'}

class PlugDPLFormatter:
    def __init__(self, config, mode, *args, **params):
        self.ctx_len = config.getint("train", "ctx_len")
        self.que_len = config.getint("train", "que_len")
        self.ans_len = config.getint("train", "ans_len")
        self.mlm_ratio = config.getfloat("train", "mlm_ratio")
        self.mlm_mean_len = config.getint("train", "mlm_mean_len")

        self.mode = mode
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"), "tokenizer"))


    def pading(self, tokens, max_len, pad_token_id):
        if len(tokens) < max_len:
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            tokens += [pad_token_id] * (max_len - len(tokens))
        else:
            mask = [1] * max_len
            tokens = tokens[:max_len]
        return tokens, mask

    def random_spans_noise_mask(self, length, noisy_density=0.15, mean_noise_span_length=3.0):
        num_noise_tokens = round(length * noisy_density)
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def random_segment(seq_length, num_segment):
            x = (torch.arange(seq_length - 1) < (num_segment - 1)).long()
            a = torch.randperm(seq_length - 1)
            x = x[a]
            x = F.pad(x, [1, 0])
            segment_id = torch.cumsum(x, dim=0)
            segment_lengths = torch.zeros(num_segment, dtype=torch.long).scatter_add_(0, segment_id, torch.ones(seq_length, dtype=torch.long))

            return segment_lengths

        noise_span_lengths = random_segment(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segment(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = torch.stack([nonnoise_span_lengths, noise_span_lengths], dim=1).view(num_noise_spans * 2)
        span_start_ends = torch.cumsum(interleaved_span_lengths, dim=0).view(-1, 2)
        return span_start_ends.tolist()

    def split_sent(self, doc):
        return sent_tokenize(doc)


    def continuation(self, sents):
        stokens = [self.tokenizer.encode(s, add_special_tokens=False) for s in sents]
        if len(stokens) < 3:
            ctxs = " ".join(sents)
            query = []
            ans = []
            return query, ans, ctxs

        length = 0
        for i in range(len(stokens)):
            length += len(stokens[i])
            if length > self.ctx_len + self.que_len:
                break
        stokens = stokens[:max(i, 3)]
        
        sid = random.choice(list(range(len(stokens) - 2)))
        sentinel = self.tokenizer.encode("<extra_id_0>", add_special_tokens=False)
        query = stokens[sid] + sentinel
        ans = sentinel + stokens[sid + 1] + stokens[sid + 2] + [self.tokenizer.eos_token_id]

        ctxs = " ".join([s for j, s in enumerate(sents) if j < sid or j > sid + 2])

        return query, ans, ctxs

    def process(self, data):
        if "context" in data[0]:
            text = [d["context"] for d in data if len(d["context"]) > 40]
        else:
            text = [d["text"] for d in data if len(d["text"]) > 40]
        if len(text) < len(data):
            text = text + text[:len(data) - len(text)]
        text = [self.split_sent(doc) for doc in text]

        que_input_ids, que_attention_mask, labels, dec_mask, ctxs = [], [], [], [], []
        for i in range(len(text)):
            rand = random.random()
            if "qas" in data[i]:
                qa = random.choice(data[i]["qas"])
                question = qa["question"]
                answer = qa["answer"]
                qtokens = self.tokenizer.encode(question)
                anstokens = self.tokenizer.encode(answer)
                ctx = " ".join(text[i])
            elif rand > 0.3 or len(text[i]) <= 3:
                qtokens, anstokens, ctx = self.mask_important_spans(text[i])
            elif rand > 0 and len(text[i]) > 3:
                qtokens, anstokens, ctx = self.continuation(text[i])
            # else:
            #     qtokens, anstokens, ctx = self.samedoc(text[i], text[(i + 1) % len(text)])
            # else:
            #     qtokens, anstokens, ctx = self.mask_important_spans(text[i])
                # qtokens, anstokens, ctx = self.continuation(text[i])
            ctxs.append(ctx)

            inpt, inpm = self.pading(qtokens, self.que_len, self.tokenizer.pad_token_id)
            lab, labmask = self.pading(anstokens, self.ans_len, -100)
            que_input_ids.append(inpt), que_attention_mask.append(inpm)
            labels.append(lab), dec_mask.append(labmask)

        model_inputs = {
            "que_input_ids": torch.LongTensor(que_input_ids),
            "que_attention_mask": torch.LongTensor(que_attention_mask),
            "labels": torch.LongTensor(labels),
            "decoder_attention_mask": torch.LongTensor(dec_mask)
        }

        model_inputs["decoder_input_ids"] = shift_tokens_right(model_inputs["labels"], 0, 0)
        model_inputs["labels"][:,0] = -100

        model_inputs2 = self.encode_ctx_nomlm(ctxs)
        model_inputs.update(model_inputs2)

        return model_inputs

    def encode_ctx_nomlm(self, ctxs):
        ctxinp = self.tokenizer(ctxs, padding=True, truncation=True, max_length=self.ctx_len)
        model_inputs = {
            "ctx_input_ids": torch.LongTensor(ctxinp["input_ids"]),
            "ctx_attention_mask": torch.LongTensor(ctxinp["attention_mask"])
        }
        return model_inputs


    def get_candidate_span_clusters(self, stokens, max_span_length, include_sub_clusters=False, validate=True):
        token_to_indices = defaultdict(list)
        for sid, sent in enumerate(stokens):
            for i, token in enumerate(sent):
                token_to_indices[token].append((sid, i))

        recurring_spans = []
        for token, indices in token_to_indices.items():
            for i, (sidx1, tidx1) in enumerate(indices):
                for j in range(i + 1, len(indices)):
                    sidx2, tidx2 = indices[j]
                    assert sidx1 < sidx2 or (sidx1 == sidx2 and tidx1 < tidx2)

                    max_recurring_length = 1
                    for length in range(1, max_span_length):
                        if include_sub_clusters:
                            recurring_spans.append(((sidx1, tidx1), (sidx2, tidx2), length))
                        if (tidx1 + length) >= len(stokens[sidx1]) or (tidx2 + length) >= len(stokens[sidx2]) or stokens[sidx1][tidx1 + length] != stokens[sidx2][tidx2 + length]:
                            break
                        max_recurring_length += 1

                    if max_recurring_length == max_span_length or not include_sub_clusters:
                        if stokens[sidx1][tidx1 + max_recurring_length - 1].replace("▁", "").lower() in STOPWORDS and max_recurring_length > 1:
                            max_recurring_length -= 1
                        recurring_spans.append(((sidx1, tidx1), (sidx2, tidx2), max_recurring_length))

        spans_to_clusters = {}
        spans_to_representatives = {}
        for idx1, idx2, length in recurring_spans:
            first_span, second_span = (idx1[0], idx1[1], idx1[1] + length - 1), (idx2[0], idx2[1], idx2[1] + length - 1)
            if first_span in spans_to_representatives:
                if second_span not in spans_to_representatives:
                    rep = spans_to_representatives[first_span]
                    cluster = spans_to_clusters[rep]
                    cluster.append(second_span)
                    spans_to_representatives[second_span] = rep
            else:
                cluster = [first_span, second_span]
                spans_to_representatives[first_span] = first_span
                spans_to_representatives[second_span] = first_span
                spans_to_clusters[first_span] = cluster

        if validate:
            recurring_spans = [cluster for cluster in spans_to_clusters.values()
                            if self.validate_ngram(stokens, cluster[0][0], cluster[0][1], cluster[0][2] - cluster[0][1] + 1)]
        else:
            recurring_spans = spans_to_clusters.values()
        return recurring_spans


    def validate_ngram(self, stokens, sentidx, start_index, length):
        tokens = stokens[sentidx][start_index: start_index + length]
        # If the vocab at the beginning of the span is a part-of-word (##), we don't want to consider this span.
        # if vocab_word_piece[token_ids[start_index]]:
        if tokens[0][0] != "▁":
            return False

        # # If the token *after* this considered span is a part-of-word (##), we don't want to consider this span.
        # if (start_index + length) < len(tokens) and tokens[start_index + length].startswith("##"):
        #     return False

        # if any([(not tokens[idx].isalnum()) and (not tokens[idx].startswith("##")) for idx in range(length)]):
        #     return False

        # We filter out n-grams that are all stopwords (e.g. "in the", "with my", ...)
        if any([t.lower().replace("▁", "") not in STOPWORDS for t in tokens]):
            return True
        return False

    def sort_cluster_by_length(self, clusters):
        ret = clusters
        ret.sort(key=lambda x:x[0][2] - x[0][1], reverse=True)
        return ret[:15]

    def mask_important_spans(self, sents):
        stokens = [self.tokenizer.tokenize(s) for s in sents]
        clusters = self.sort_cluster_by_length(self.get_candidate_span_clusters(stokens, 10))
        sid2span = defaultdict(list)
        for cluster in clusters:
            for span in cluster:
                sid2span[span[0]].append(span)

        sids = random.sample(list(sid2span.keys()), min(5, len(sid2span)))
        sids.sort()

        query, ans, real_selected = [], [], []
        maskid = 0
        for sid in sids:
            real_selected.append(sid)
            sspans = sid2span[sid]
            sspans.sort(key=lambda x:x[1] * 100 + x[2])
            mask_span = [list(sspans[0])]
            for s in sspans[1:]:
                if s[1] <= mask_span[-1][2]:
                    mask_span[-1][1] = min(s[1], mask_span[-1][1])
                    mask_span[-1][2] = max(s[2], mask_span[-1][2])
                else:
                    mask_span.append(list(s))
            begin = 0
            qtoken = []
            for s in mask_span:
                sentinel = ["<extra_id_%s>" % maskid]
                qtoken.extend(stokens[sid][begin : s[1]] + sentinel)
                ans.extend(sentinel + stokens[sid][s[1] : s[2] + 1])

                maskid += 1
                begin = s[2] + 1
                if len(qtoken) > len(stokens[sid]) * 0.7:
                    break
            qtoken.extend(stokens[sid][begin:])
            qtoken.append("*")
            query.extend(qtoken)
            if len(query) > self.que_len:
                break
        query.append("<extra_id_%s>" % maskid)
        ans.append("<extra_id_%s>" % maskid)
        query = self.tokenizer.convert_tokens_to_ids(query)
        ans = self.tokenizer.convert_tokens_to_ids(ans)
        ctxs = " ".join([s for j, s in enumerate(sents) if not j in real_selected])

        return query, ans, ctxs


