from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class ChunkedDataset(Dataset):
    def __init__(self, stream_generator, chunk_size, max_size, base_index=0, pool_size=3, load_size=1):
        # pool_size表示窗口大小
        # load_size表示触发时加载时，一次性加载几个chunk(也意味着移除前面的几个chunk)
        # max_size的作用只是方便外面的index传递多少个而已，实际可以一直读取
        # 但是sampler通过访问len(dataset)就只读取那么多个
        # 所以len(dataset)非常重要，也就是max_size,想要完整的数据集，可以将max_size设置成数据集最大行数
        # 注意因为数据集最大行数不一定是chunk_size的整数倍，所以最后一个chunk可能数据不满足
        self.max_size = max_size
        self.index_offset = 0
        self.data_pool = []
        self.chunk_sizes = []
        self.chunk_size = chunk_size
        self.stream_generator = stream_generator
        self.end_index = 0
        i = 0
        for data_chunk in self.stream_generator:
            # data_chunk is a list of data
            self.data_pool += data_chunk
            self.chunk_sizes.append(len(data_chunk)) 
            i += 1
            if i >= pool_size:
                break
        self.end_index = len(self.data_pool)+self.index_offset
        self.load_size = load_size

    def __len__(self, ):
        return self.max_size

    def __getitem__(self, index):
        if index >= self.end_index:
            # 触发加载机制
            self._update_pool()
            logger.debug(f"get {index}, data pool update!")
            if index >= self.end_index:
                logger.error("index change to large!!!")
        if index < self.index_offset:
            logger.warning("can't not go to back, minus index get wrong data...")
        return self.data_pool[index-self.index_offset]

    def _update_pool(self, ):
        i = 0
        tmp_pool = []
        tmp_sizes = []
        for data_chunk in self.stream_generator:
            tmp_pool += data_chunk
            tmp_sizes.append(len(data_chunk))
            i += 1
            if i >= self.load_size:
                break
        if tmp_sizes != []:
            num = len(tmp_sizes)
            remove_num = sum(self.chunk_sizes[:num])
            self.index_offset = self.index_offset + remove_num
            self.data_pool = self.data_pool[remove_num:]+tmp_pool
            self.chunk_sizes = self.chunk_sizes[num:]+tmp_sizes
            self.end_index = self.end_index + sum(tmp_sizes)
