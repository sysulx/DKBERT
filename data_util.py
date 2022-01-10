import os
import pickle
import hashlib
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset
import logging

TOKENIZED_DATA_ROOT = '/home/lx/data/ir/data'

logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.length = len(self.data_list)
    def __len__(self, ):
        return self.length
    def __getitem__(self, index):
        return self.data_list[index]

# 下面是第 1 种输入方式，query 和 doc 拼接到一起

def load_and_tokenize(file_path, tokenizer, max_len, topk, file_type='train_triples', load_type='pairwise'):
    data_list = []
    i = 0
    if file_type == 'train_triples':
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="load triples")):
                if isinstance(topk, int) and i >= topk:
                    break
                query, pos_doc, neg_doc = line.strip().split('\t')[:3]
                inputs = tokenizer(
                    query,
                    pos_doc,
                    add_special_tokens=True,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_token_type_ids=True,
                    # return_tensors="pt",
                )
                inputs2 = tokenizer(
                    query,
                    neg_doc,
                    add_special_tokens=True,
                    max_length= max_len,
                    padding='max_length',
                    truncation=True,
                    return_token_type_ids=True,
                    # return_tensors="pt",
                )
                if load_type == 'pairwise':
                    example = {
                        "input_ids": inputs['input_ids'],
                        "attention_mask": inputs['attention_mask'],
                        "token_type_ids": inputs['token_type_ids'],
                        "input_ids2": inputs2['input_ids'],
                        "attention_mask2": inputs2['attention_mask'],
                        "token_type_ids2": inputs2['token_type_ids'],
                        "label": 1, # inputs1 is positive, inputs2 is negative
                    }
                    data_list.append(example)
                else:
                    example1 = {
                        "input_ids": inputs['input_ids'],
                        "attention_mask": inputs['attention_mask'],
                        "token_type_ids": inputs['token_type_ids'],
                        "label":1,
                    }
                    example2 = {
                        "input_ids": inputs2['input_ids'],
                        "attention_mask": inputs2['attention_mask'],
                        "token_type_ids": inputs2['token_type_ids'],
                        "label":0,
                    }
                    data_list.append(example1)
                    data_list.append(example2)
    else:
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="load triples")):
                if isinstance(topk, int) and i >= topk:
                    break
                query, doc, label = line.strip().split('\t')[:3]
                inputs = tokenizer(
                    query,
                    doc,
                    add_special_tokens=True,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_token_type_ids=True,
                    # return_tensors="pt",
                )
                example = {
                    "input_ids": inputs['input_ids'],
                    "attention_mask": inputs['attention_mask'],
                    "token_type_ids": inputs['token_type_ids'],
                    "label": int(label),
                }
                data_list.append(example)
    return data_list, i

def get_pairwise_tokenized_datalist(file_path, tokenizer, max_len=256, topk='ALL'):
    file_name = os.path.split(file_path)[1]
    hashcode = hashlib.md5(file_name.encode('utf-8')).hexdigest() 
    namestr = os.path.join(TOKENIZED_DATA_ROOT, f"./tokenized_data_list.1.{hashcode}.{max_len}.{topk}.pairwise.pkl")
    if not os.path.exists(namestr):
        data_list, i = load_and_tokenize(file_path, tokenizer, max_len, topk)
        logger.info(f"read and tokenize {i} lines...")
        logger.info(f"save data to {namestr}")
        pickle.dump(data_list, open(namestr, 'wb'))
    else:
        logger.info(f"load data from {namestr}")
        data_list = pickle.load(open(namestr, 'rb'))
    logger.info(f"data length:{len(data_list)}")
    logger.info(f"data keys:{data_list[0].keys()}")
    return data_list

def get_pointwise_tokenized_datalist(file_path, tokenizer, max_len=256, topk='ALL', file_type='query_doc_label'):
    file_name = os.path.split(file_path)[1]
    hashcode = hashlib.md5(file_name.encode('utf-8')).hexdigest() 
    namestr = os.path.join(TOKENIZED_DATA_ROOT, f"./tokenized_data_list.1.{hashcode}.{max_len}.{topk}.pointwise.pkl")
    if not os.path.exists(namestr):
        data_list, i = load_and_tokenize(file_path, tokenizer, max_len, topk, file_type=file_type, load_type='pointwise')
        logger.info(f"read and tokenize {i} lines...")
        logger.info(f"save data to {namestr}")
        pickle.dump(data_list, open(namestr, 'wb'))
    else:
        logger.info(f"load data from {namestr}")
        data_list = pickle.load(open(namestr, 'rb'))
    logger.info(f"data length:{len(data_list)}")
    logger.info(f"data keys:{data_list[0].keys()}")
    return data_list


# 下面是第 2 种输入方式，query 和 doc 不进行拼接

def _tokenize_and_cache(text, tokenizer, cache, max_len):
    if text in cache:
        return cache[text]
    tmp = tokenizer(text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        # return_tensors="pt",
    )
    cache[text] = tmp
    return tmp

def single_load_and_tokenize(file_path, tokenizer, max_len, topk, file_type='train_triples', load_type='pairwise',query_max_len=None,doc_max_len=None):
    data_list = []
    if query_max_len is None:
        query_max_len = max_len
    if doc_max_len is None:
        doc_max_len = max_len
    i = 0
    query_cache = {}
    doc_cache = {}
    if file_type == 'train_triples':
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="load triples")):
                if isinstance(topk, int) and i >= topk:
                    break
                query, pos_doc, neg_doc = line.strip().split('\t')[:3]
                query_input = _tokenize_and_cache(query, tokenizer, query_cache, query_max_len)
                doc1_input = _tokenize_and_cache(pos_doc, tokenizer, doc_cache, doc_max_len)
                doc2_input = _tokenize_and_cache(neg_doc, tokenizer, doc_cache, doc_max_len)
                
                if load_type == 'pairwise':
                    example = {
                        "query_input_ids": query_input['input_ids'],
                        "query_attention_mask": query_input['attention_mask'],
                        "query_token_type_ids": query_input['token_type_ids'],
                        "doc_input_ids": doc1_input['input_ids'],
                        "doc_attention_mask": doc1_input['attention_mask'],
                        "doc_token_type_ids": doc1_input['token_type_ids'],
                        "doc2_input_ids": doc2_input['input_ids'],
                        "doc2_attention_mask": doc2_input['attention_mask'], 
                        "doc2_token_type_ids": doc2_input['token_type_ids'],
                        "label": 1, # inputs1 is positive, inputs2 is negative
                    }
                    data_list.append(example)
                else:
                    example1 = {
                        "query_input_ids": query_input['input_ids'],
                        "query_attention_mask": query_input['attention_mask'],
                        "query_token_type_ids": query_input['token_type_ids'],
                        "doc_input_ids": doc1_input['input_ids'],
                        "doc_attention_mask": doc1_input['attention_mask'],
                        "doc_token_type_ids": doc1_input['token_type_ids'],
                        "label":1,
                    }
                    example2 = {
                        "query_input_ids": query_input['input_ids'],
                        "query_attention_mask": query_input['attention_mask'],
                        "query_token_type_ids": query_input['token_type_ids'],
                        "doc_input_ids": doc2_input['input_ids'],
                        "doc_attention_mask": doc2_input['attention_mask'],
                        "doc_token_type_ids": doc2_input['token_type_ids'],
                        "label":0,
                    }
                    data_list.append(example1)
                    data_list.append(example2)
    else:
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="load query doc labels")):
                if isinstance(topk, int) and i >= topk:
                    break
                query, doc, label = line.strip().split('\t')[:3]
                query_input = _tokenize_and_cache(query, tokenizer, query_cache, query_max_len)
                doc_input = _tokenize_and_cache(doc, tokenizer, doc_cache, doc_max_len)
                example = {
                    "query_input_ids": query_input['input_ids'],
                    "query_attention_mask": query_input['attention_mask'],
                    "query_token_type_ids": query_input['token_type_ids'],
                    "doc_input_ids": doc_input['input_ids'],
                    "doc_attention_mask": doc_input['attention_mask'],
                    "doc_token_type_ids": doc_input['token_type_ids'],
                    "label": int(label),
                }
                data_list.append(example)
    return data_list, i

def get_pointwise_single_tokenized_datalist(file_path, tokenizer, max_len=256, topk='ALL', file_type='query_doc_label', query_max_len=None, doc_max_len=None):
    file_name = os.path.split(file_path)[1]
    hashcode = hashlib.md5(file_name.encode('utf-8')).hexdigest() 
    query_max_len = query_max_len or max_len 
    doc_max_len = doc_max_len or max_len
    namestr = os.path.join(TOKENIZED_DATA_ROOT, f"./tokenized_data_list.1.{hashcode}.{query_max_len}.{doc_max_len}.{topk}.pointwise.single.pkl")
    if not os.path.exists(namestr):
        data_list, i = single_load_and_tokenize(file_path, tokenizer, max_len, topk, file_type=file_type, load_type='pointwise', 
            query_max_len=query_max_len, doc_max_len=doc_max_len)
        logger.info(f"read and tokenize {i} lines...")
        logger.info(f"save data to {namestr}")
        pickle.dump(data_list, open(namestr, 'wb'))
    else:
        logger.info(f"load data from {namestr}")
        data_list = pickle.load(open(namestr, 'rb'))
    logger.info(f"data length:{len(data_list)}")
    logger.info(f"data keys:{data_list[0].keys()}")
    return data_list

def get_pairwise_single_tokenized_datalist(file_path, tokenizer, max_len=256, topk='ALL', query_max_len=None, doc_max_len=None):
    file_name = os.path.split(file_path)[1]
    hashcode = hashlib.md5(file_name.encode('utf-8')).hexdigest() 
    query_max_len = query_max_len or max_len 
    doc_max_len = doc_max_len or max_len
    # [TODO] 修改下面的保存路径{query_max_len}.{doc_max_len}
    namestr = os.path.join(TOKENIZED_DATA_ROOT, f"./tokenized_data_list.1.{hashcode}.{query_max_len}.{doc_max_len}.{topk}.pairwise.single.pkl")
    if not os.path.exists(namestr):
        data_list, i = single_load_and_tokenize(file_path, tokenizer, max_len, topk, 
        query_max_len=query_max_len, doc_max_len=doc_max_len)
        logger.info(f"read and tokenize {i} lines...")
        logger.info(f"save data to {namestr}")
        pickle.dump(data_list, open(namestr, 'wb'))
    else:
        logger.info(f"load data from {namestr}")
        data_list = pickle.load(open(namestr, 'rb'))
    logger.info(f"data length:{len(data_list)}")
    logger.info(f"data keys:{data_list[0].keys()}")
    return data_list

def get_single_dataset(file_path, tokenizer, max_len, topk, file_type='query_doc_label', mode="pointwise", query_max_len=64, doc_max_len=256):
    if mode == 'pointwise':
        data_list = get_pointwise_single_tokenized_datalist(file_path, tokenizer, max_len, topk, file_type,query_max_len=query_max_len, doc_max_len=doc_max_len)
    elif mode == 'pairwise':
        data_list = get_pairwise_single_tokenized_datalist(file_path, tokenizer, max_len, topk, query_max_len=query_max_len, doc_max_len=doc_max_len)
    return SimpleDataset(data_list)


import torch
def collate_fn_fix(features, 
    float_ks = set(("query_attention_mask", "doc_attention_mask", "doc2_attention_mask", "attention_mask")),
    long_ks = set(("query_input_ids", "query_token_type_ids", "doc_input_ids", "doc2_input_ids", "doc_token_type_ids", "doc2_token_type_ids", "labels","input_ids","token_type_ids",'label'))
    
    ):

    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    elif "labels" in first and first['labels'] is not None:
        label = first["labels"].item() if isinstance(first["labels"], torch.Tensor) else first["labels"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=dtype)
    
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                if k in float_ks:
                    dtype = torch.float
                else:
                    dtype = torch.long
                batch[k] = torch.tensor([f[k] for f in features], dtype=dtype)
    if "query_token_type_ids" not in batch:
        length = len(features)
        batch['query_token_type_ids'] = torch.zeros((length, 64), dtype=torch.long)
        batch['doc_token_type_ids'] = torch.zeros((length, 256), dtype=torch.long)
        if "doc2_input_ids" in batch:
            batch['doc2_token_type_ids'] = torch.zeros((length, 256), dtype=torch.long)
    return batch

