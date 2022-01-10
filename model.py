import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import math
from collections import abc
from numbers import Number
import numpy as np
from transformers.modeling_outputs import ModelOutput
from knrm_model import KNRMTopLayer2

from transformers import BertPreTrainedModel, BertModel, BertTokenizer, BertLayer, BertConfig
#from src.parameters import DEVICE
from copy import deepcopy
logger = logging.getLogger(__name__)

def get_repr_str(value, depth=1):
    if depth > 0:
        if isinstance(value, ModelOutput):
            return value.__class__.__name__+f"({','.join(k+':'+get_repr_str(v, depth-1) for k,v in value.items())})"
        elif isinstance(value, abc.Mapping):
            return str({k:get_repr_str(v, depth-1) for k, v in value.items()})
        elif isinstance(value, list):
            return str([get_repr_str(v,depth-1) for v in value])
        elif isinstance(value, tuple):
            return str(tuple(get_repr_str(v,depth-1) for v in value))
    if isinstance(value, Number):
        return str(value)
    elif isinstance(value, torch.Tensor):
        size = value.size()
        if len(size) == 0:
            return str(value)
        return str(value.size())
    else:
        return str(type(value))#value.__class__.__name__

def log_tensor_shape(var_dict, only=None, exclude=None, log_times=None, trace=[0]):
    "配合locals打印函数局部变量(tensor类型)的信息"
    if log_times is not None and log_times <= trace[0]:
        return
    strs = ["-"*10+f" logger vars{trace[0]} "+"-"*10]
    for name, value in sorted(var_dict.items()):
        # if not isinstance(value, torch.Tensor):
        #     logger.info(f"var:{name} type:{type(value)}")
        #     continue
        if only is not None:
            if name in only:
                strs.append("variable:\t{:32s}".format(name) + get_repr_str(value))
        elif exclude is not None:
            if name not in exclude:
                strs.append("variable:\t{:32s}".format(name) + get_repr_str(value))
        else:
            strs.append("variable:\t{:32s}".format(name)+ get_repr_str(value))
    strs.append("")
    logger.info("\n".join(strs))
    trace[0] += 1 

# def focal_loss_func(pos_score, neg_score, labels, fl_gamma=2):
#     diff = (pos_score-neg_score) # shape: (B, 1)
#     probs = torch.sigmoid(diff) # shape:(B, 1)
#     return -((1-probs)**fl_gamma*torch.log(probs)).sum()

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # torch.arange return a torch.LongTensor but it should be a float tensor
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*-(math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DBERT(BertPreTrainedModel):
    def __init__(self, config,):
        super(DBERT, self).__init__(config)
        self.config = config
        self.query_maxlen = config.query_maxlen
        self.doc_maxlen = config.doc_maxlen
        # cosine or l2 两种计算相似度的方法 
        # self.similarity_metric = config.similarity_metric 

        doc_bert_path = os.environ['BERT_BASE_UNCASED']
        query_bert_path = os.environ['BERT_BASE_UNCASED']
        if hasattr(config, "doc_bert_path") and config.doc_bert_path is not None:
            doc_bert_path = config.doc_bert_path
        if hasattr(config, "query_bert_path") and config.query_bert_path is not None:
            query_bert_path = config.query_bert_path
        self.doc_bert = BertModel.from_pretrained(doc_bert_path)
        if config.only_doc_bert == True:
            self.query_bert = self.doc_bert
        else:
            self.query_bert = BertModel.from_pretrained(query_bert_path)
        
        if hasattr(config, 'fix_bert_param') and config.fix_bert_param == True:
            logger.warn("fixed bert parameters, only train the top layer...")
            for param in self.query_bert.parameters():
                param.requires_grad = False
            for param in self.doc_bert.parameters():
                param.requires_grad = False
        
        
        self.linear1 = nn.Linear(self.config.hidden_size, self.config.TOP_hidden_size)
        # self.query_proj = self.linear1
        # self.doc_proj = self.linear1
        self.query_proj = nn.Linear(self.config.hidden_size, self.config.TOP_hidden_size)
        self.doc_proj = nn.Linear(self.config.hidden_size, self.config.TOP_hidden_size)
        
        
        if hasattr(config, 'fix_colbert_param') and config.fix_colbert_param == True:
            logger.warn("fixed colbert parameters, only train the transformer layer...")
            for param in self.query_bert.parameters():
                param.requires_grad = False
            for param in self.doc_bert.parameters():
                param.requires_grad = False
            for param in self.linear1.parameters():
                param.requires_grad = False
            for param in self.query_proj.parameters():
                param.requires_grad = False
            for param in self.doc_proj.parameters():
                param.requires_grad = False
        
        self.pe = PositionalEncoding(d_model=self.config.TOP_hidden_size, dropout=0.1, max_len=512)
        
        top_config = self.create_config()
        self.top_layer = BertLayer(top_config)
        
        self.output_layer = nn.Linear(in_features=top_config.TOP_hidden_size, out_features=1)
        # self.loss_func = focal_loss_func
        self.loss_func = lambda pos_score, neg_score, labels:-torch.log(torch.sigmoid(pos_score-neg_score)).sum()
        # self.loss_func = nn.CrossEntropyLoss()
        # def loss_func(pos_score, neg_score, labels):
        #     return F.relu(neg_score-pos_score).sum()
        # self.loss_func =loss_func
        
        # 这个是方便Trianer这个框架使用的，需要有个loss占据返回值的第一个位置
        self.register_buffer('tmp_loss', torch.Tensor(1)) # for trainer
        
        self.knrm = KNRMTopLayer2(config)
        for param in self.knrm.parameters():
            param.requires_grad = False

        self.log_times = 1

        self.alpha = config.alpha

    def create_config(self, ):
        "替换掉TOP_的配置项，防止和底层BERT冲突"
        top_config = deepcopy(self.config)
        tmp = []
        for k, v in self.config.__dict__.items():
            if k.startswith('TOP_'):
                top_config.__dict__[k[4:]] = v
                tmp.append(k[4:])
        tmp = {k:getattr(top_config, k) for k in tmp}
        logger.warning(f"top_config: {tmp}")
        return top_config
    
    def make_top_input_and_mask(self, query_attention_mask, query_outputs, 
            doc_attention_mask, doc_outputs,
        ):
        attention_mask = torch.cat((query_attention_mask[:,:], doc_attention_mask[:,:]),dim=1)
        inputs = torch.cat((query_outputs[:,:,:], doc_outputs[:,:,:]),dim=1)
        input_shape = inputs.size()
        device = attention_mask.device
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        return extended_attention_mask, inputs

    def similarity_match(self, query_attention_mask, query_squeeze_norm, 
            doc_attention_mask, doc_squeeze_norm, 
        ):
        similarity_matrix = torch.matmul(query_squeeze_norm[:,1:], doc_squeeze_norm.permute(0,2,1))
        batch_size = query_attention_mask.size(0)
        query_mask = query_attention_mask[:,1:].float() # 去掉[CLS]
        doc_mask = doc_attention_mask.float()
        # matrix_mask = torch.matmul(query_mask.unsqueeze(2), doc_mask.unsqueeze(1))
        # similarity_matrix.masked_fill_(matrix_mask==0, value=-9999)
        # query_aspect: batch_size * query_len
        query_max_sim = similarity_matrix.max(2).values
        # query_max_sim.masked_fill_(query_mask==0.0, value=0.0)
        query_based_score = query_max_sim.sum(dim=-1,keepdims=True)
        return query_based_score, similarity_matrix, query_mask, doc_mask

    def transformer_layer(self, query_input, doc_input, query_mask, doc_mask):
        query_input = self.pe(query_input)
        doc_input = self.pe(doc_input)
        attention_mask, top_inputs = self.make_top_input_and_mask(query_mask, query_input, doc_mask, doc_input)
        t_outputs = self.top_layer(hidden_states=top_inputs, attention_mask=attention_mask, output_attentions=False)
        # [batch_size, q_len+doc_len, hidden_dim]
        return t_outputs[0]
        
    def forward(self, query_input_ids, query_attention_mask, query_token_type_ids,  
            doc_input_ids, doc_attention_mask, doc_token_type_ids,
            doc2_input_ids=None, doc2_attention_mask=None, doc2_token_type_ids=None, labels=None,
            **kwargs):
        query_len = query_input_ids.size(1)
        query_input_ids.masked_fill_(query_attention_mask==0, value=103)
        query_output = self.query_bert(query_input_ids, query_attention_mask, query_token_type_ids)
        doc_output = self.doc_bert(doc_input_ids, doc_attention_mask, doc_token_type_ids)
        
        # query_input = self.query_proj(query_output.last_hidden_state)
        # doc_input = self.doc_proj(doc_output.last_hidden_state)
        
        query_input = self.linear1(query_output.last_hidden_state)
        doc_input = self.linear1(doc_output.last_hidden_state)

        query_squeeze_norm = F.normalize(query_input, p=2, dim=-1)
        doc_squeeze_norm = F.normalize(doc_input, p=2, dim=-1)
        
        # score1_1 = self.similarity_match(query_attention_mask, query_squeeze_norm, 
        #     doc_attention_mask, doc_squeeze_norm, 
        # )

        # # one transformer layer
        # t_outputs = self.transformer_layer(query_input, doc_input, query_attention_mask, doc_attention_mask)
        # t_q_outputs_norm = F.normalize(t_outputs[:, :query_len], p=2, dim=-1)
        # t_doc_outputs_norm = F.normalize(t_outputs[:, query_len:], p=2,dim=-1)

        # # concatenate
        # query_squeeze_norm_cat = torch.cat((query_squeeze_norm, t_q_outputs_norm), dim=-1)
        # doc_squeeze_norm_cat = torch.cat((doc_squeeze_norm, t_doc_outputs_norm), dim=-1)
        
        score1_1, similarity_matrix, query_mask, doc_mask = self.similarity_match(query_attention_mask, query_squeeze_norm,
            doc_attention_mask, doc_squeeze_norm,
        )
        # print(score1_1.size())
        # print(self.knrm(similarity_matrix).size())
        # print(score1_1)
        # print(self.knrm(similarity_matrix))
        # zzz
        score1 = score1_1 /63 + torch.tanh(self.knrm(similarity_matrix, query_mask, doc_mask))
        # score1 = torch.tanh(self.knrm(similarity_matrix, query_mask, doc_mask))
        # score1 = self.knrm(similarity_matrix)

        if doc2_input_ids is not None:
            # labels1 = torch.ones((score1.size(0),), dtype=torch.long, device=score1.device)
            # loss1 = self.loss_func(score1, labels1)
            doc2_output = self.doc_bert(doc2_input_ids, doc2_attention_mask, doc2_token_type_ids)
            # doc2_input = self.doc_proj(doc2_output.last_hidden_state)
            doc2_input = self.linear1(doc2_output.last_hidden_state)

            doc2_squeeze_norm = F.normalize(doc2_input, p=2, dim=-1)

            # score2_1 = self.similarity_match(query_attention_mask, query_squeeze_norm, 
            #     doc2_attention_mask, doc2_squeeze_norm, 
            # )

            # # one transformer layer
            # t_outputs2 = self.transformer_layer(query_input, doc2_input, query_attention_mask, doc2_attention_mask)
            # t_q_outputs_norm2 = F.normalize(t_outputs2[:, :query_len], p=2, dim=-1)
            # t_doc_outputs_norm2 = F.normalize(t_outputs2[:, query_len:], p=2,dim=-1)

            # query2_squeeze_norm_cat = torch.cat((query_squeeze_norm, t_q_outputs_norm2), dim=-1)
            # doc2_squeeze_norm_cat = torch.cat((doc2_squeeze_norm, t_doc_outputs_norm2), dim=-1)

            score2_1,  similarity_matrix, query_mask, doc_mask = self.similarity_match(query_attention_mask, query_squeeze_norm, 
                doc2_attention_mask, doc2_squeeze_norm,
            )

            score2 = score2_1 /63 +torch.tanh(self.knrm(similarity_matrix, query_mask, doc_mask))
            # score2 = torch.tanh(self.knrm(similarity_matrix, query_mask, doc_mask))
            loss = self.loss_func(score1, score2, 1)
        else:
            loss = self.tmp_loss

        if self.log_times > 0:
            log_tensor_shape(locals(), exclude=["self",])
            self.log_times -= 1
        return loss, score1

    def get_model_output(self,  query_input_ids, query_attention_mask, query_token_type_ids,  
            doc_input_ids, doc_attention_mask, doc_token_type_ids,
            doc2_input_ids=None, doc2_attention_mask=None, doc2_token_type_ids=None, labels=None,
            **kwargs):
        query_len = query_input_ids.size(1)
        query_input_ids.masked_fill_(query_attention_mask==0, value=103)
        query_output = self.query_bert(query_input_ids, query_attention_mask, query_token_type_ids)
        doc_output = self.doc_bert(doc_input_ids, doc_attention_mask, doc_token_type_ids)
        
        query_input = self.linear1(query_output.last_hidden_state)
        doc_input = self.linear1(doc_output.last_hidden_state)

        query_squeeze_norm = F.normalize(query_input, p=2, dim=-1)
        doc_squeeze_norm = F.normalize(doc_input, p=2, dim=-1)
       
        score1_1, similarity_matrix, query_mask, doc_mask = self.similarity_match(query_attention_mask, query_squeeze_norm,
            doc_attention_mask, doc_squeeze_norm,
        )
        
        out, phi = self.knrm.get_model_output(similarity_matrix, query_mask, doc_mask)

        score1 = score1_1/63 + torch.tanh(self.knrm(similarity_matrix, query_mask, doc_mask))
        return score1, score1_1/63, out, phi
