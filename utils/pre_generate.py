from stream_util import get_pairwise_tokenized_datalist_stream, TOKENIZED_DATA_ROOT
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



triples_file = os.path.join(TOKENIZED_DATA_ROOT, 'triples.train.small.tsv.each_query_first_10_record')
tokenizer = BertTokenizerFast.from_pretrained(os.environ['BERT_BASE_UNCASED'])

# # chunk_size = 1 太慢了 8个小时都没搞完一半
# topk = 'ALL'
# generator = get_pairwise_tokenized_datalist_stream(triples_file, tokenizer, topk=topk, chunk_size=1, num_workers=20, max_size=200)
# count = 0
# for item in generator:
#     if count < 1:
#         logger.info(type(item))
#         try:
#             logger.info(item.keys())
#         except:
#             logger.error(f"single item error! not dict")
#         logger.info(str(item)[:100]+"...")
#     count += 1
# logger.info(f"items count is {count}")
# if topk != count:
#     logger.warn(f"if the file has more than {topk} rows, there may be something wrong")
# else:
#     logger.info("item mode testing passed!")
# logger.info("finished!")

# chunk_size 调的越大，后面加载数据的速度就越快！

topk = 'ALL'
chunk_size=100000 # 10000
generator = get_pairwise_tokenized_datalist_stream(triples_file, tokenizer, topk=topk, chunk_size=chunk_size, num_workers=20, max_size=100,)
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
