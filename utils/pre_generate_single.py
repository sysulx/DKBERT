from single_stream_util import get_train_triples_pairwise_tokenized_datalist_stream, TOKENIZED_DATA_ROOT, get_eval_qdl_pointwise_tokenized_datalist_stream
import os
from transformers import BertTokenizerFast
import logging
import math
import pickle
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s[line:%(lineno)d] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

file_name = 'triples.train.small.tsv.each_query_first_10_record'
file_name = 'top1000.dev.q_doc_label.pad.list.txt'
file_path = os.path.join(TOKENIZED_DATA_ROOT, file_name)
tokenizer = BertTokenizerFast.from_pretrained(os.environ['BERT_BASE_UNCASED'])

topk = 'ALL'
chunk_size = 20000 # 10000
# generator = get_train_triples_pairwise_tokenized_datalist_stream(file_path, tokenizer, topk=topk, chunk_size=chunk_size, num_workers=20, max_size=100,)
generator = get_eval_qdl_pointwise_tokenized_datalist_stream(file_path, tokenizer, topk=topk, chunk_size=chunk_size, num_workers=20, max_size=100,)
logger.info("tokenize and save finished, begin testing...")
count = 0
num = 0
for item in tqdm(generator):
    if count < 1:
        logger.info(type(item))
        if len(item) != chunk_size:
            logger.error(f"chunk stream error! not list")
        try:
            logger.info(item[0].keys())
        except:
            logger.error(f"single item error! not dict")
        logger.info(str(item[0])[:100]+"...")
    count += 1
    num += len(item)
logger.info(f"chunks count is {count}")
logger.info(f"num examples is {num}")
if isinstance(topk, int) and math.ceil(topk/chunk_size) != count:
    logger.warn(f"if the file have more than {topk} rows, there may be something wrong")
else:
    logger.info("chunk mode testing passed!")
