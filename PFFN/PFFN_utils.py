import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List
from labml import tracker

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# PositionalEncoding (from Transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512): #input_encoder_dim, sequence_length, batch_size,
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe = pe[:sequence_length, :32].unsqueeze(0).repeat(batch_size, 1, 1)
        #self.pe = pe
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        #x = x + self.pe[:x.size(0), :].squeeze(1)
        # print(x.size())
        return x


def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, -1)


def Encoder_a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, -1)


# attention

def attention(Q, K, V):
    a = a_norm(Q, K)
    return torch.matmul(a, V)


def Encoder_attention(Q, K, V):
    a = Encoder_a_norm(Q, K)
    return torch.matmul(a, V)


# Query, Key and Value

class Key(torch.nn.Module):
    def __init__(self, input_encoder_dim, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.input_encoder_dim = input_encoder_dim
        self.fc1 = nn.Linear(32, dim_attn, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Query(torch.nn.Module):
    def __init__(self, input_encoder_dim, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        self.input_encoder_dim = input_encoder_dim
        self.fc1 = nn.Linear(32, dim_attn, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Value(torch.nn.Module):
    def __init__(self, input_encoder_dim, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.input_encoder_dim = input_encoder_dim
        self.fc1 = nn.Linear(32, dim_val, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


# AttentionBlock
class Attention(torch.nn.Module):
    def __init__(self, input_encoder_dim, dim_attn):
        super(Attention, self).__init__()
        self.value = Value(input_encoder_dim, input_encoder_dim)
        self.key = Key(input_encoder_dim, dim_attn)
        self.query = Query(input_encoder_dim, dim_attn)

    def forward(self, x, kv=None):
        if (kv is None):
            return attention(self.query(x), self.key(x), self.value(x))
        return attention(self.query(x), self.key(kv), self.value(kv))


class Encoder_Attention(torch.nn.Module):
    def __init__(self, input_encoder_dim, dim_attn):
        super(Encoder_Attention, self).__init__()
        self.value = Value(input_encoder_dim, input_encoder_dim)
        self.key = Key(input_encoder_dim, dim_attn)
        self.query = Query(input_encoder_dim, dim_attn)

    def forward(self, x, kv=None):
        if (kv is None):
            return Encoder_attention(self.query(x), self.key(x), self.value(x))
        return Encoder_attention(self.query(x), self.key(kv), self.value(kv))


# Multi-head self-attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_encoder_dim, dim_attn, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(Attention(input_encoder_dim, dim_attn))
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads * input_encoder_dim, input_encoder_dim, bias=False)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))
        a = torch.stack(a, dim=-1)
        a = a.flatten(start_dim=2)
        x = self.fc(a)
        return x


class Encoder_MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_encoder_dim, dim_attn, n_heads):
        super(Encoder_MultiHeadAttention, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(Encoder_Attention(input_encoder_dim, dim_attn))
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads * input_encoder_dim, 32, bias=False)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))
        a = torch.stack(a, dim=-1)
        a = a.flatten(start_dim=2)
        x = self.fc(a)
        return x


class AveragedModel(torch.nn.Module):
    def __init__(self, model):
        super(AveragedModel, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
