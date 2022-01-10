import logging
import os
import hashlib
from tqdm import tqdm
import pickle
from .multiprocess_util import Distributed
from torch.utils.data import Dataset

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

def file_read_generator(file_path, topk):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if isinstance(topk, int) and i >= topk:
                break
            yield line.strip().split('\t')[:3]

def data_stream_read_generator(file_path, topk='ALL'):
    with open(file_path,'rb') as f:
        count = 0
        while True:
            try:
                item = pickle.load(f)
                yield item
                count += 1
                if isinstance(topk, int) and count >= topk:
                    break
            except EOFError:
                break
 
def tokenize_lines(line_generator, tokenizer, max_len, file_type='train_triples', load_type='pairwise'):
    data_list = []
    if file_type == 'train_triples':
        for query, pos_doc, neg_doc in line_generator:
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
        for query, doc, label in line_generator:
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
    return data_list

def get_stream_handler(file_path):
    def stream_handler(stream):
        with open(file_path, 'ab') as f:
            num_chunks = 0
            for chunk in stream:
                pickle.dump(chunk, f)
                num_chunks += 1
        return num_chunks
    return stream_handler

def get_pairwise_tokenized_datalist_stream(file_path, tokenizer, root_path=TOKENIZED_DATA_ROOT, max_len=256, chunk_size=1, topk='ALL', num_workers=10, max_size=100):
    file_name = os.path.split(file_path)[1]
    hashcode = hashlib.md5(file_name.encode('utf-8')).hexdigest() 
    namestr = os.path.join(root_path, f"./tokenized_data_list.1.{hashcode}.{max_len}.{topk}.{chunk_size}.pairwise.pkl.multi")
    if not os.path.exists(namestr):
        source_stream = file_read_generator(file_path, topk=topk)
        distributed = Distributed(num_workers=num_workers, max_size=max_size)
        stream_handler = get_stream_handler(namestr)
        if chunk_size == 1:
            logger.warning("data source will save as a list of item, not a list of list...")
            decorator_fn = distributed.multiprocess(map_keys='line_generator', chunk_size=1, stream_handler=stream_handler, time_out=1200)
        else:
            decorator_fn = distributed.multiprocess(map_keys='line_generator', chunk_size=chunk_size, stream=False, stream_handler=stream_handler, time_out=1200)
        tokenize_stream_fn = decorator_fn(tokenize_lines)
        num_chunks = tokenize_stream_fn(source_stream, tokenizer, max_len)
        logger.info(f"Totally {num_chunks} data have been saved..")
    logger.info(f"return data stream generator from {namestr}")
    return data_stream_read_generator(namestr)

