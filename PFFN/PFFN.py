import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import numpy as np
from data_process_for_model import extract_local_fea
from torch.utils.data import Dataset
from PFFN_utils import *
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.sgdr import CosineAnnealingLR_with_Restart
from torch.optim import Adam, AdamW
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from metrics import score

device = torch.device("cpu")

class MyDataset(Dataset):
    def __init__(self, x, x_att, y):
        super(MyDataset, self).__init__()
        self.len = x.shape[0]
        device = 'cpu'
        self.x_encoder_data = torch.as_tensor(x, device='cpu', dtype=torch.float)
        self.x_att_data = torch.as_tensor(x_att, device='cpu', dtype=torch.float)
        self.y_data = torch.as_tensor(y, device='cpu', dtype=torch.float)

    def __getitem__(self, index):
        return self.x_encoder_data[index], self.x_att_data[index], self.y_data[index]
    def __len__(self):
        return self.len

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_encoder_dim, hidden_dim, dim_attn_s, n_heads=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = Encoder_MultiHeadAttention(input_encoder_dim, dim_attn_s, n_heads)
        self.fc1 = nn.Linear(32, self.hidden_dim)

        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        a = self.attn(src)
        fea_encoder = self.norm1(src + a)
        a = self.fc1(F.elu(self.fc1(fea_encoder)))
        fea_encoder = self.norm2(fea_encoder + a)
        return fea_encoder


class att_EncoderLayer(torch.nn.Module):
    def __init__(self,input_att_dim, hidden_dim, num_heads=1, dim_attn=8, dropout=0.1):
        super(att_EncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_layer_att = nn.Dropout(dropout)
        self.attn = MultiheadAttention(input_att_dim, dim_attn, num_heads)
        self.fc1 = nn.Linear(input_att_dim, dim_attn)
        self.fc2 = nn.Linear(dim_attn, input_att_dim)
        self.norm1 = nn.LayerNorm(input_att_dim)
        self.norm2 = nn.LayerNorm(input_att_dim)
        self.input_att_dim = input_att_dim

    def forward(self, src_att):
        #fea_fc = self.dropout_layer_fc(self.fc(src_fc))
        a, _ = self.attn(src_att, src_att, src_att)
        fea_att = self.norm1(src_att + a)
        a = self.fc1(F.elu(self.fc2(fea_att)))
        fea_att = self.norm2(fea_att + a)
        fea_att = self.dropout_layer_att(fea_att)
        return fea_att

class PFFN(torch.nn.Module):
    def __init__(self, input_encoder_dim, input_att_dim, hidden_dim, dim_attn_s=32, dim_val=32, n_encoder_layers=1, n_heads=1):

        super(PFFN, self).__init__()
        #self.dec_seq_len = dec_seq_len

        # Initiate Encoder encoder
        self.encoder_encoder = []
        for i in range(n_encoder_layers):
            self.encoder_encoder.append(EncoderLayer(input_encoder_dim, dim_attn_s, n_heads))

        # Initiate Fully Connected encoder
        self.att_encoder = []
        self.att_encoder.append(att_EncoderLayer(input_att_dim, hidden_dim))
        self.att_encoder = nn.ModuleList(self.att_encoder)

        self.pos_s = PositionalEncoding(dim_attn_s) #input_encoder_dim, Sequence_len, batch_size
        self.att_enc_input_fc = nn.Linear(input_att_dim, input_att_dim)
        self.encoder_enc_input_fc = nn.Linear(dim_attn_s, dim_attn_s)
        self.norm1 = nn.LayerNorm(dim_val, dim_val)
        self.hidden_dim = hidden_dim
        self.fc2 = nn.Linear(3200, 16)
        self.flatten = nn.Flatten()

        self.regressor = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim * 2, out_features=1),
        )
    def forward(self, fea_encoder,fea_att):

        # input embedding and positional encoding

        encoder_x = fea_encoder.transpose(1, 2)

        e = self.encoder_encoder[0](self.pos_s(self.encoder_enc_input_fc(encoder_x)))  # ((batch_size,sensor,dim_val_s))
        o= self.att_encoder[0](self.att_enc_input_fc(fea_att))  # ((batch_size,timestep,dim_val_t))

        # sensors encoder
        for encoder_enc in self.encoder_encoder[1:]:
            e = encoder_enc(e)

        # fc encoder
        for att_enc in self.att_encoder[1:]:
            o = att_enc(o)

        e = self.flatten(e)
        e = self.fc2(e)

        # feature fusion
        p = torch.cat((e, o), dim=1)
        x = self.regressor(p)
        return x

epochs = 4000
lr_min = 1e-6
hidden_dim = 12
iterations = 5
t_max = 200
t_mult = 1
snapshot_em_start = 10


model_saved_dir = './result'
data_path = './processed_data/first_100_cycle_data_hybrid.pt'

data_set = torch.load(data_path)
train_x_encoder = torch.from_numpy(data_set['train_x_encoder']).permute(0, 2, 1).float()
train_y = torch.from_numpy(data_set['train_y']).float()
train_x_att = torch.from_numpy(data_set['train_x_att']).float()

train_x_eva_encoder = torch.from_numpy(data_set['eva_x_encoder']).permute(0, 2, 1).float()
train_y_eva = torch.from_numpy(data_set['eva_y']).float()
train_x_eva_att = torch.from_numpy(data_set['eva_x_att']).float()

test_x_pri_encoder = torch.from_numpy(data_set['test_x_pri_encoder']).permute(0, 2, 1).float()
test_y_pri = torch.from_numpy((data_set['test_y_pri'])).float()
test_x_pri_att = torch.from_numpy(data_set['test_x_pri_att']).float()

test_x_sec_encoder = torch.from_numpy(data_set['test_x_sec_encoder']).permute(0, 2, 1).float()
test_y_sec = torch.from_numpy(data_set['test_y_sec']).float()
test_x_sec_att = torch.from_numpy(data_set['test_x_sec_att']).float()

print("train_x_encoder shape ={}, train_y shape ={}, train_x_att shape={}".format(train_x_encoder.shape, train_y.shape, train_x_att.shape))
print("train_x_eva_encoder shape ={}, train_y_eva shape ={}, train_x_eva_att shape={}".format(train_x_eva_encoder.shape, train_y_eva.shape, train_x_eva_att.shape))
print("test_x_pri_encoder shape ={}, test_y_pri shape ={}, test_x_pri_att shape={}".format(test_x_pri_encoder.shape, test_y_pri.shape, test_x_pri_att.shape))
print("test_x_sec_encoder shape ={}, test_y_sec shape ={}, test_x_sec_att shape={}".format(test_x_sec_encoder.shape, test_y_sec.shape, test_x_sec_att.shape))

train_rmse_snapshot, train_err_snapshot = [], []
test_pri_rmse_snapshot, test_pri_err_snapshot, test_pri_scores_snapshot = [], [], []
test_sec_rmse_snapshot, test_sec_err_snapshot, test_sec_scores_snapshot = [], [], []
test_pri_pred_snapshot = []
test_sec_pred_snapshot = []

train_enable = True

if train_enable:

    model = PFFN(input_encoder_dim=train_x_encoder.shape[1], input_att_dim=train_x_att.shape[1], hidden_dim=hidden_dim).to('cpu') #batch_size=train_x_encoder.shape[0]

    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001, momentum=0.85)

    criterion = nn.MSELoss()
    mySGDR = CosineAnnealingLR_with_Restart(optimizer=optimizer, T_max=t_max, T_mult=t_mult, model=model,out_dir=model_saved_dir,take_snapshot=True, eta_min=lr_min)
    losses = []
    for epoch in range(1,epochs+1):
        mySGDR.step()
        model.train()
        Loss = 0.0
        train_x_att_ = train_x_att.to('cpu')
        train_x_encoder_ = train_x_encoder.to('cpu')
        train_y_ = train_y.to('cpu')
        optimizer.zero_grad()
        pred = model(train_x_encoder_,train_x_att_)
        loss = criterion(pred, train_y_)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        Loss += loss.item()
        losses.append(loss.item())

        if epoch % 100 ==0:
            model.eval()
            with torch.no_grad():
                train_x_eva_ = train_x_eva_encoder.to('cpu')
                train_x_eva_att_ = train_x_eva_att.to('cpu')
                pred = model(train_x_eva_, train_x_eva_att_)
                pred_ = pred.to('cpu').numpy()
                pred_ = np.power(10, pred_)
                train_y_eva_ = np.power(10, train_y_eva)
                rmse = sqrt(mean_squared_error(train_y_eva_, pred_))
            print("Epoch[{}/{}] | Train Loss = {:.5f}, Train RMSE={:.2f}".format(epoch, epochs, Loss, rmse))
import glob

checkpoints = sorted(glob.glob(model_saved_dir + '/*.tar'))

models = []

for path in checkpoints:
    model = PFFN(input_encoder_dim=train_x_encoder.shape[1], input_att_dim=train_x_att.shape[1], hidden_dim=hidden_dim).to('cpu') #batch_size=train_x_encoder.shape[0]
    ch = torch.load(path)
    model.load_state_dict(ch['state_dict'])
    models.append(model)

i = 0
train_rmse, train_err = [], []
test_pri_rmse, test_pri_err, test_pri_scores = [], [], []
test_sec_rmse, test_sec_err, test_sec_scores= [], [], []

for model in models[snapshot_em_start:]:
    i+=1
    model.eval()
    with torch.no_grad():
        train_x_eva_encoder_ = train_x_eva_encoder.to('cpu')
        train_x_eva_att_ = train_x_eva_att.to('cpu')
        y_pred = model(train_x_eva_encoder_,train_x_eva_att_)
        y_pred = y_pred.to('cpu').numpy()
        # y_pred = scaler.inverse_transform(y_pred)
        y_pred = np.power(10, y_pred)
        train_y_eva_ = np.power(10, train_y_eva)
        #train_y_eva_.append(train_y_eva_)
        rmse = sqrt(mean_squared_error(train_y_eva_,y_pred))
        err = np.average(np.divide(np.abs(train_y_eva_-y_pred),train_y_eva_)) *100
        print("Snapshot-{} Prediction | Training RMSE = {:.2f}, Error ={:.2f}".format(i, rmse,err))
        train_rmse.append(rmse)
        train_err.append(err)
        

        test_x_pri_encoder_= test_x_pri_encoder.to('cpu')
        test_x_pri_att_ = test_x_pri_att.to('cpu')
        y_pri_pred = model(test_x_pri_encoder_,test_x_pri_att_)
        y_pri_pred = y_pri_pred.to('cpu').numpy()
        # y_pred = scaler.inverse_transform(y_pred)
        y_pri_pred = np.power(10, y_pri_pred)
        test_pri_pred = y_pri_pred
        rmse = sqrt(mean_squared_error(test_y_pri,y_pri_pred))
        err = np.average(np.divide(np.abs(test_y_pri-y_pri_pred),test_y_pri)) *100
        scores = score(y_pri_pred, test_y_pri)
        print("Snapshot-{} Prediction | Primary Test RMSE = {:.2f}, Error = {:.2f}, Scores = {:.2f}".format(i, rmse,err,scores))
        test_pri_rmse.append(rmse)
        test_pri_err.append(err)
        test_pri_scores.append(scores)

        test_x_sec_encoder_ = test_x_sec_encoder.to('cpu')
        test_x_sec_att_ = test_x_sec_att.to('cpu')
        y_sec_pred = model(test_x_sec_encoder_,test_x_sec_att_)
        y_sec_pred = y_sec_pred.to('cpu').numpy()
        # y_pred = scaler.inverse_transform(y_pred)
        y_sec_pred = np.power(10, y_sec_pred)
        test_sec_pred = y_sec_pred
        rmse = sqrt(mean_squared_error(test_y_sec,y_sec_pred))
        err = np.average(np.divide(np.abs(test_y_sec-y_sec_pred),test_y_sec)) *100
        scores = score(y_sec_pred, test_y_sec)
        print("Snapshot-{} Prediction | Secondary RMSE = {:.2f}, Error ={:.2f}, Scores = {:.2f}".format(i, rmse,err,scores))
        test_sec_rmse.append(rmse)
        test_sec_err.append(err)
        test_sec_scores.append(scores)
        print()
    train_rmse_snapshot.append(sum(train_rmse) / len(train_rmse))
    train_err_snapshot.append(sum(train_err) / len(train_err))

    test_pri_rmse_snapshot.append(sum(test_pri_rmse) / len(test_pri_rmse))
    test_pri_err_snapshot.append(sum(test_pri_err) / len(test_pri_err))
    test_pri_scores_snapshot.append(sum(test_pri_scores) / len(test_pri_scores))
    #test_pri_pred_snapshot.append(sum(test_pri_y_pred) / len(test_pri_y_pred))

    test_sec_rmse_snapshot.append(sum(test_sec_rmse) / len(test_sec_rmse))
    test_sec_err_snapshot.append(sum(test_sec_err) / len(test_sec_err))
    test_sec_scores_snapshot.append(sum(test_sec_scores) / len(test_sec_scores))
    #test_sec_pred_snapshot.append(sum(test_sec_y_pred) / len(test_sec_y_pred))
print()

print()
print("Snapshot | Average Training RMSE={:.2f}({:.2f}), Error={:.2f}({:.2f})".format(
    sum(train_rmse_snapshot) / len(train_rmse_snapshot),
    np.std(train_rmse_snapshot),
    sum(train_err_snapshot) / len(train_err_snapshot),
    np.std(train_err_snapshot)))
print("Snapshot | Average Primary Test RMSE={:.2f}({:.2f}), Error={:.2f}({:.2f}), Scores={:.2f}({:.2f})".format(
    sum(test_pri_rmse_snapshot) / len(test_pri_rmse_snapshot),
    np.std(test_pri_rmse_snapshot),
    sum(test_pri_err_snapshot) / len(test_pri_err_snapshot),
    np.std(test_pri_err_snapshot),
    sum(test_pri_scores_snapshot) / len(test_pri_scores_snapshot),
    np.std(test_pri_scores_snapshot)))
print("Snapshot | Average Secondary Test RMSE={:.2f}({:.2f}), Error={:.2f}({:.2f}), Scores={:.2f}({:.2f})".format(
    sum(test_sec_rmse_snapshot) / len(test_sec_rmse_snapshot),
    np.std(test_sec_rmse_snapshot),
    sum(test_sec_err_snapshot) / len(test_sec_err_snapshot),
    np.std(test_sec_err_snapshot),
    sum(test_sec_scores_snapshot) / len(test_sec_scores_snapshot),
    np.std(test_sec_scores_snapshot)))
print("End")


#Bayesian Optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Define the search space for hyperparameters
space = {
    'n_heads': hp.choice('n_heads', [1, 2, 4, 8]),

    # Add more hyperparameters to optimize if needed
}

def objective(params):
    # Create the model with the current set of hyperparameters
    model = PFFN(input_encoder_dim=train_x_encoder.shape[1], input_att_dim=train_x_att.shape[1], hidden_dim=hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001, momentum=0.85)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(1, epochs + 1):
        mySGDR.step()
        model.train()
        Loss = 0.0
        optimizer.zero_grad()
        pred = model(train_x_encoder_, train_x_att_)
        loss = criterion(pred, train_y_)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        Loss += loss.item()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_x_eva_ = train_x_eva_encoder.to('cpu')
        train_x_eva_att_ = train_x_eva_att.to('cpu')
        pred = model(train_x_eva_, train_x_eva_att_)
        pred_ = pred.to('cpu').numpy()
        pred_ = np.power(10, pred_)
        train_y_eva_ = np.power(10, train_y_eva)
        rmse = sqrt(mean_squared_error(train_y_eva_, pred_))

    return {'loss': rmse, 'status': STATUS_OK}

# Run the Bayesian optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best hyperparameters:")
print(best)
