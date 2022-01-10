from transformers import Trainer as Trans_Trainer
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import get_tpu_sampler
from transformers.optimization import get_constant_schedule, AdamW
from .sampler import ChunkedDistributedSampler
from .chunked_dataset import ChunkedDataset
import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import collections
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)
SEED = 42 # 和 trainer的default一致
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def get_lr_func(self):
    def f(num_training_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_constant_schedule(self.optimizer)
    return f

class Trainer(Trans_Trainer):
    """
    在原有transformers.Trainer类的基础上:
        1. 在训练时支持分布式分块数据集的Sampler:ChunkedDistributedSampler
        2. 支持以非分布式情况下，运行ChunkedDataset和ChunkedDistributedSampler
        3. 支持常数级的学习率，即在训练过程中不对学习率进行变动

    """
    def __init__(self, *args, fix_lr=False, **kwargs):
        super().__init__(*args, **kwargs)
        if fix_lr:
            logger.warn("training with fixed learning-rate!")
            # 设置常数学习率
            self.create_optimizer_and_scheduler = get_lr_func(self)
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            if isinstance(self.train_dataset, ChunkedDataset):
                if self.args.local_rank == -1: # for non-distributed testing
                    return ChunkedDistributedSampler(self.train_dataset, shuffle=False, num_replicas=10, rank=0)
                return ChunkedDistributedSampler(self.train_dataset, shuffle=True)
            if self.args.local_rank == -1:
                return RandomSampler(self.train_dataset)
            else:
                return DistributedSampler(self.train_dataset)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )
