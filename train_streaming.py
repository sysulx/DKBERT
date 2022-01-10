import faulthandler
faulthandler.enable()
import os
import pickle
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizerFast, BertModel, PreTrainedModel, BertConfig, TrainingArguments
from transformers.file_utils import ModelOutput
from transformers.trainer_utils import is_main_process
from typing import Optional, List, Any, Dict
import logging
from data_util import get_single_dataset,collate_fn_fix
from model import DBERT
from utils.trainer import Trainer
from utils.single_stream_util import data_stream_read_generator
from utils.chunked_dataset import ChunkedDataset
from wiki_dataset import WikiPointwiseDataset, WikiPairwiseDataset

SEED = 42 # 和 trainer的default一致
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TOKENIZED_DATA_ROOT = '/home/lx/data/ir/data'

logger = logging.getLogger(__name__)

TRAIN_FILE = '/home/lx/data/ir/data/million_train.txt'
DEV_FILE = '/home/lx/data/ir/data/top1000.dev.top1000.txt'
MAX_LENGTH = 256
train_topk = 1000000 #1000000
dev_topk = 1000

# this file is use for hard dev examples
dynamic_dev_file = "/home/lx/data/ir/data/hard_dev_example.tsv"
tokenizer = BertTokenizerFast.from_pretrained(os.environ['BERT_BASE_UNCASED'])

def mrr_closure(dynamic_dev_file=None, num_doc_per_query=1000, k=10):
    "@param dynamic_dev_file: query\t doc\t label \t ..."
    "@param num_doc_per_query: int or 'dynamic'"
    "@param k: topk of MRR, MRR@K"
    def generate_query_ids(rawtxt_filename):
        query_ids = []
        tmp_dict = {}
        num = 0
        with open(rawtxt_filename) as f:
            for line in f.readlines():
                query = line.split("\t")[0] # rawtxt_filename每一行的第一个元素是query
                if query not in tmp_dict:
                    tmp_dict[query] = num
                    num += 1
                query_ids.append(tmp_dict[query])
            return query_ids
    query_ids = []
    if dynamic_dev_file is not None:
        num_doc_per_query = 'dynamic'
        query_ids = generate_query_ids(dynamic_dev_file)
    def get_mrr_at_k(labels, probs):
        # labels: 1-d numpy array
        # probs: 1-d numpy array, it can also be relevance scores.
        # the dev set must be sorted by query id, all doc respected to one query must be the consecutive part 
        if num_doc_per_query == 'dynamic':
            # 表示动态扩展，数目不定，但是按照query连续进来的，同一个query排在一起
            # 但是 metric不允许有额外的输入，所以需要全局变量 query 标记
            all_ = np.stack((labels, probs),axis=1)
            group_data = {}
            for i,qid in enumerate(query_ids):
                if qid not in group_data:
                    group_data[qid] = []
                group_data[qid].append(all_[i])
            query_groups = list(group_data.values())
        else:
            total = len(labels)
            assert total == len(probs)
            assert total%num_doc_per_query == 0
            all_ = np.stack((labels, probs),axis=1) 
            query_groups = [all_[i*num_doc_per_query:(i+1)*num_doc_per_query] for i in range(int(total/num_doc_per_query))]
        mrr = 0
        for group in query_groups:
            ranked = sorted(group, key=lambda r: float(r[-1]), reverse=True)
            for i in range(min(len(ranked), k)):
                if ranked[i][0] == 1:
                    mrr += 1.0/(i+1)
                    break
        mrr = mrr/len(query_groups)
        return mrr
    return get_mrr_at_k

def compute_metrics_closure(mrr_func):
    def compute_metrics(pred):
        labels = pred.label_ids
        num_dims = len(pred.predictions.shape)
        if num_dims == 1:
            scores = pred.predictions
        elif pred.predictions.shape[1] == 1: # single output units, but without squeeze, just for a relevance score
            scores = pred.predictions[:,0]
        else: # two output units, logits for label==0 and label==1
            if pred.predictions.shape[1] != 2 or num_dims != 2:
                logger.error(f"the evaluation may be fault! pred.predictions.shape{pred.predictions.shape}")
            # preds = pred.predictions.argmax(-1)
            preds_exp = np.exp(pred.predictions)
            preds_exp_sum = np.sum(preds_exp, axis=1, keepdims=True)
            scores = preds_exp/preds_exp_sum
            scores = scores[:,1] # probs for true relevance
        mrr = mrr_func(labels, scores)
        metrics = {'mrr':mrr}
        return metrics
    return compute_metrics

if __name__=='__main__':
    # load data
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if is_main_process(args.local_rank) else logging.WARN,
        format="%(asctime)s %(levelname)-8s %(name)s[line:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    # dev_dataset = get_dataset(file_path=DEV_FILE, tokenizer=tokenizer, 
    #     max_len=MAX_LENGTH, topk=dev_topk)
    # mrr_func = mrr_closure(dynamic_dev_file=None)
    query_max_len = 64
    doc_max_len =256
    # dev_dataset = get_single_dataset(file_path=dynamic_dev_file, tokenizer=tokenizer, 
    #     max_len=MAX_LENGTH, topk='all',
    #     query_max_len=query_max_len, doc_max_len=doc_max_len)
    wiki_doc2ids = pickle.load(open("/home/lx/data/ir/data/wikiqa/tokenized_data/wiki_docid2ids.pkl",'rb'))
    wiki_q2ids = pickle.load(open("/home/lx/data/ir/data/wikiqa/tokenized_data/wiki_qid2ids.pkl",'rb'))
    
    wiki_dev_tsv = "/home/lx/data/ir/data/wikiqa/wikiqacorpus/WikiQA-dev.tsv"
    wiki_dev_txt = "/home/lx/data/ir/data/wikiqa/wikiqacorpus/WikiQA-dev.txt"
    dev_dataset = WikiPointwiseDataset(wiki_dev_tsv, wiki_doc2ids, wiki_q2ids)

    wiki_test_tsv = "/home/lx/data/ir/data/wikiqa/wikiqacorpus/WikiQA-test.tsv"
    wiki_test_txt = "/home/lx/data/ir/data/wikiqa/wikiqacorpus/WikiQA-test.txt"
    test_dataset = WikiPointwiseDataset(wiki_test_tsv, wiki_doc2ids, wiki_q2ids)

    wiki_train_tsv = "/home/lx/data/ir/data/wikiqa/wikiqacorpus/WikiQA-train.tsv"
    wiki_train_txt = "/home/lx/data/ir/data/wikiqa/wikiqacorpus/WikiQA-train.txt"
    train_pair_dataset = WikiPairwiseDataset(wiki_train_tsv, wiki_doc2ids, wiki_q2ids)

    mrr_func = mrr_closure(dynamic_dev_file=wiki_test_txt)
    
    # TRAIN_CHUNKS = 195 # 最大195表示全量数据
    # TRAIN_FILE_CHUNKED = '/home/lx/data/ir/data/tokenized_data_list.1.9035670fdb2b8fde1e397ab2f2d61915.64.256.ALL.20000.single.pairwise.pkl.multi'
    # CHUNK_SIZE = 20000
    # TRAIN_DATA_SIZE = 3892205 #这个是上限
    # MAX_SIZE = min(TRAIN_DATA_SIZE, TRAIN_CHUNKS*CHUNK_SIZE)
    # stream_generator = data_stream_read_generator(TRAIN_FILE_CHUNKED, topk=TRAIN_CHUNKS)
    # train_dataset = ChunkedDataset(stream_generator, chunk_size=CHUNK_SIZE, max_size=MAX_SIZE, base_index=0, pool_size=3, load_size=1)

    config = BertConfig.from_dict({
        "query_maxlen": query_max_len,
        "doc_maxlen": doc_max_len,
        "only_doc_bert": True,
        "query_bert_path":os.environ['BERT_BASE_UNCASED'], #"/home/lx/data/ir/pretrain/query/best_ckpt/",
        "doc_bert_path": "/home/lx/data/ir/pretrain/doc/best_ckpt/",
        "fix_bert_param": False,
        "TOP_hidden_size": 128,
        "TOP_num_attention_heads": 4,
        "TOP_intermediate_size": 128*4,
        "TOP_hidden_dropout_prob": 0.1,
        "alpha": 0.5, # 0 <= alpha <= 1
        "fix_colbert_param":False,
    })
    training_args = TrainingArguments(
        learning_rate=1e-4,
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=64,
        warmup_steps=20,
        weight_decay=0.01,
        logging_steps=1*5,
        save_steps= 5*5,
        eval_steps= 5*5,
        gradient_accumulation_steps=3*5,
        evaluation_strategy = 'steps',
        local_rank=args.local_rank, # 重要，否则是DP
        seed=SEED,
        logging_first_step=True,
    )
    model = DBERT(config=config)
    # 注意覆盖掉config，不然仅仅加载checkpoints的config，上面的config不会生效。
    # model = DBERT.from_pretrained('/home/lx/data/ir/interactive2/n1/t5/results/checkpoint-7500/', config=config)
    # model.load_state_dict(torch.load('/home/lx/data/ir/interactive/v8/t1/results/checkpoint-105000/pytorch_model.bin', map_location='cpu'), strict=False)
    # 已经修改了kernel的个数，使用from_pretrained会因为网络架构不一样而出现错误。此时应该选择上一种创建方法。
    model = DBERT.from_pretrained("../t6/results/checkpoint-200/", config=config)
    # weights = torch.load('../t4/results/checkpoint-350/pytorch_model.bin', map_location='cpu')
    # del weights['knrm.out.weight']
    # del weights['knrm.out.bias']
    # model.load_state_dict(weights, strict=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_pair_dataset,
        eval_dataset=test_dataset,
        compute_metrics = compute_metrics_closure(mrr_func),
        data_collator = collate_fn_fix,
        fix_lr = False,
    )
    print(trainer.evaluate())
    # if args.local_rank <= 0:
    #     torch.save(model.state_dict(), "./model.init.pkl")
    trainer.train()

