import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s[line:%(lineno)d] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
)
def pad_one_seq(qids, pad_len):
    qids = qids[:pad_len-2] #完成对qids的拷贝，后面不会真的修改外面传入的qids
    qids.insert(0, 101)
    qids.append(102)
    actual_length = len(qids)
    qids += [0]*(pad_len-actual_length)
    return qids, actual_length

def pad_batch_seq(batch_seqids, pad_len=None):
    if pad_len is None:
        pad_len = max(len(r)+2 for r in batch_seqids)
    batch_ids = []
    batch_act_lens = []
    for seqids in batch_seqids:
        pad_ids, actual_length = pad_batch_seq(seqids, pad_len)
        batch_ids.append(pad_ids)
        batch_act_lens.append(actual_length)
    return batch_ids, batch_act_lens

def read_wiki_tsv(file_name, skip_first=True):
    with open(file_name) as f:
        data = f.read().strip().split("\n")
    logger.info(f"{file_name} has {len(data)} examples")
    if skip_first:
        data = data[1:]
    for r in tqdm(data):
        tmp = r.split('\t')
        yield [tmp[0], tmp[4], int(tmp[-1])]


class WikiPointwiseDataset(Dataset):
    def __init__(self, file_name, doc2ids, q2ids, preload=True, q_max_len=64, doc_max_len=256):
        data = list(read_wiki_tsv(file_name))
        self.length = len(data)
        if preload:
            self.datalist = self.pre_compute(data, q_max_len, doc_max_len, doc2ids, q2ids)
        else:
            # self.doc2ids = doc2ids
            # self.q2ids = q2ids
            self.init_load(data, doc2ids, q2ids)
        
    def init_load(self, data, doc2ids, q2ids):
        datalist = []
        for r in tqdm(data):
            qids = q2ids[r[0]]
            docids = doc2ids[r[1]]
            datalist.append({
                "query_input_ids": qids,
                "doc_input_ids": docids,
                "labels": r[2],
            })
        self.datalist = datalist

    def pre_compute(self, data, q_max_len, doc_max_len,doc2ids,q2ids):
        datalist = []
        len2mask = np.tril(np.ones(doc_max_len, dtype=np.float32)) 
        all_token_type_ids = np.zeros(doc_max_len, dtype=np.int32)
        for r in tqdm(data):
            qids = q2ids[r[0]]
            docids = doc2ids[r[1]]
            pad_qids, act_q_len = pad_one_seq(qids, q_max_len)
            pad_docids, act_doc_len = pad_one_seq(docids, doc_max_len)
            datalist.append({
                "query_input_ids": pad_qids,
                "query_attention_mask": len2mask[act_q_len-1][:q_max_len],
                "query_token_type_ids": all_token_type_ids[:q_max_len],
                "doc_input_ids": pad_docids,
                "doc_attention_mask": len2mask[act_doc_len-1][:doc_max_len],
                "doc_token_type_ids": all_token_type_ids[:doc_max_len],
                "labels": r[2],
            })
        return datalist

    def __getitem__(self, index):
        return self.datalist[index]
    
    def __len__(self,):
        return self.length

def generate_pairwise_data(data):
    # data: list of [qid, docid, label]
    group_by_qid = {}
    for r in data:
        group_by_qid.setdefault(r[0],[]).append(r)
    
    all_docs = set()
    for group in group_by_qid.values():
        for r in group:
            all_docs.add(r[1])
    all_docs = list(all_docs)
    all_length = len(all_docs)
    pairs = []
    count_non_rel = 0
    count_rel = 0
    for group in group_by_qid.values():
        cls_group = {0:[],1:[]}
        for r in group:
            cls_group[r[-1]].append(r)
        if len(cls_group[0]) == 0:
            count_non_rel += 1
            # logger.warn(f"qid {r[0]} has no nonrelevant docs.")
        if len(cls_group[1]) == 0:
            count_rel += 1
            # logger.warn(f"qid {r[0]} has no relevant docs.")
        for r1 in cls_group[1]:
            for r2 in cls_group[0]:
                pairs.append([r1[0], r1[1], r2[1]])
            length = len(cls_group[0])
            if length < 10:
                random_docs = np.random.choice(range(all_length), 10-length, replace=False)
                for idx in random_docs:
                    pairs.append([r1[0],r1[1],all_docs[idx]])
    logger.warning(f"{count_non_rel} qid have no negtive docs.")
    logger.warning(f"{count_rel} qid have no positive docs.")
    return pairs



class WikiPairwiseDataset(Dataset):
    def __init__(self, file_name, doc2ids, q2ids, preload=True, q_max_len=64, doc_max_len=256):
        data = list(read_wiki_tsv(file_name))
        data = generate_pairwise_data(data)
        logger.info(f"{file_name} has {len(data)} pairs.")
        self.length = len(data)
        if preload:
            self.datalist = self.pre_compute(data, q_max_len, doc_max_len, doc2ids, q2ids)
        else:
            # self.doc2ids = doc2ids
            # self.q2ids = q2ids
            self.init_load(data, doc2ids, q2ids)
        
    def init_load(self, data, doc2ids, q2ids):
        datalist = []
        for r in tqdm(data):
            qids = q2ids[r[0]]
            docids = doc2ids[r[1]]
            docids2 = doc2ids[r[2]]
            datalist.append({
                "query_input_ids": qids,
                "doc_input_ids": docids,
                "doc2_input_ids": docids2,
                "labels": 1,
            })
        self.datalist = datalist

    def pre_compute(self, data, q_max_len, doc_max_len,doc2ids,q2ids):
        datalist = []
        len2mask = np.tril(np.ones(doc_max_len, dtype=np.float32)) 
        all_token_type_ids = np.zeros(doc_max_len, dtype=np.int32)
        for r in tqdm(data):
            qids = q2ids[r[0]]
            docids = doc2ids[r[1]]
            docids2 = doc2ids[r[2]]
            pad_qids, act_q_len = pad_one_seq(qids, q_max_len)
            pad_docids, act_doc_len = pad_one_seq(docids, doc_max_len)
            pad_docids2, act_doc_len2 = pad_one_seq(docids2, doc_max_len)
            datalist.append({
                "query_input_ids": pad_qids,
                "query_attention_mask": len2mask[act_q_len-1][:q_max_len],
                "query_token_type_ids": all_token_type_ids[:q_max_len],
                "doc_input_ids": pad_docids,
                "doc_attention_mask": len2mask[act_doc_len-1][:doc_max_len],
                "doc_token_type_ids": all_token_type_ids[:doc_max_len],
                "doc2_input_ids": pad_docids2,
                "doc2_attention_mask": len2mask[act_doc_len2-1][:doc_max_len],
                "doc2_token_type_ids": all_token_type_ids[:doc_max_len],
                "labels": 1,
            })
        return datalist

    def __getitem__(self, index):
        return self.datalist[index]
    
    def __len__(self,):
        return self.length

