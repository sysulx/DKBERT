import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianKernel(nn.Module):
    """
    Gaussian kernel module.

    :param mu: Float, mean of the kernel.
    :param sigma: Float, sigma of the kernel.

    Examples:
        >>> import torch
        >>> kernel = GaussianKernel()
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> kernel(x).shape
        torch.Size([4, 5, 10])
    """
    def __init__(self, mu: float = 1., sigma: float = 1.):
        """Gaussian kernel constructor."""
        super().__init__()
        self.mu = mu
        self.sigma = sigma
    def forward(self, x):
        """Forward."""
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )

class KNRMTopLayer2(nn.Module):
    """
    KNRM Model.

    Examples:
        >>> model = KNRM()
        >>> model.params['kernel_num'] = 11
        >>> model.params['sigma'] = 0.1
        >>> model.params['exact_sigma'] = 0.001
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()
    """
    @classmethod
    def get_default_params(cls):
        return {"kernel_num":11, "sigma":0.1, "exact_sigma":0.001,}

    def __init__(self, config):
        super().__init__()
        params = KNRMTopLayer2.get_default_params()
        for k in params.keys():
            if hasattr(config, k):
                params[k] = config.k
        self._params = params
        self.build()
    def build(self, ):
        """build model structure"""
        self.kernels = nn.ModuleList()
        for i in range(self._params['kernel_num']):
            # mu = 1. / (self._params['kernel_num'] - 1) + (2. * i) / (
            #     self._params['kernel_num'] - 1) - 1.0
            # sigma = self._params['sigma']
            # if mu > 1.0:
            #     sigma = self._params['exact_sigma']
            #     mu = 1.0
            mu = i / (self._params['kernel_num'] - 1)
            sigma = self._params['sigma']
            if mu >= 1.0:
                sigma = self._params['exact_sigma']
            self.kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        self.out = nn.Linear(self._params['kernel_num'], 1)

    def forward(self, matching_matrix, query_mask, doc_mask, **kwargs):
        """Forward."""
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   K = number of kernels

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        # query, doc = inputs['text_left'], inputs['text_right']

        # Process left input.
        # shape = [B, L, D]
        # embed_query = self.embedding(query.long())
        # shape = [B, R, D]
        # embed_doc = self.embedding(doc.long())

        # shape = [B, L, R]
        # matching_matrix = torch.einsum(
            # 'bld,brd->blr',
            # F.normalize(embed_query, p=2, dim=-1),
            # F.normalize(embed_doc, p=2, dim=-1)
        # )
        out, phi = self.get_model_output(matching_matrix, query_mask, doc_mask, **kwargs)

        return out
    
    def get_model_output(self, matching_matrix, query_mask, doc_mask, **kwargs):
        """
        matching_matrix: B, Q_num, D_num
        query_mask: B, Q_num
        doc_mask: B, D_num
        """


        KM = []
        doc_mask = doc_mask.unsqueeze(1)
        for kernel in self.kernels:
            # # shape = [B]
            # if matching_mask is None:
            #     K = torch.log1p(kernel(matching_matrix).max(dim=-1).values).sum(dim=-1)
            # else:
            #     K = torch.log1p((kernel(matching_matrix)*matching_mask).max(dim=-1).values).sum(dim=-1)
            # # K = torch.log1p(kernel(matching_matrix).max(dim=-1).values).sum(dim=-1)
            #K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            K = (torch.log1p((kernel(matching_matrix)*doc_mask).sum(dim=-1))*query_mask).sum(dim=-1)
            
            KM.append(K)

        # shape = [B, K]
        phi = torch.stack(KM, dim=1)

        out = self.out(phi) # KNRM代码实现里面好像没有激活函数，论文里面有tanh
        # out = torch.tanh(out)
        # out = torch.sigmoid(out)
        return out, phi

class AttenCross(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # self.atten_config.TOP_hidden_size
    def forward(self, query_input, query_mask, doc_input, doc_mask, sim_matrix):
        """
        *_input: (batch, seq_len, hidden)
        *_mask: (batch, seq_len)
        sim_matrix: (batch, q_seq_len, doc_seq_len)
        # 如果需要去掉query的[CLS]输出，请在调用函数时，采用切片出入参数，这里不做任何处理
        """
        batch_size, q_len, dim = query_input.size()
        # cross_atten_mat: (batch, q_seq_len, doc_seq_len)
        cross_atten_mat = torch.matmul(query_input, doc_input.permute(0,2,1))/np.sqrt(dim)
        doc_mat_mask = doc_mask.unsqueeze(1).expand(-1, q_len, -1)
        cross_atten_mat.masked_fill_(doc_mat_mask==0, value=-9999.0)
        cross_atten_mat = F.softmax(cross_atten_mat, dim=-1)
        query_atten_sim = (cross_atten_mat*sim_matrix).sum(dim=-1)
        return query_atten_sim.sum(dim=-1,keepdim=True)
