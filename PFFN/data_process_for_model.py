import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pickle
import os
import torch
from sklearn import preprocessing
import joblib
import argparse
from statistics import variance
from scipy.stats import skew,kurtosis,pearsonr
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from feature_selection import extract_delta_Q_variance, extract_bat_cycle_life,extract_fea_for_hybrid_model
from sklearn.metrics import auc

def cycle_feat_extract(x):
    x_data = np.arange(len(x)) + 1
    Qdlin_min = np.min(x)
    Qdlin_max = np.max(x)
    Qdlin_mean = np.mean(x)
    Qdlin_var = np.var(x)
    Qdlin_median = np.median(x)                #中位数
    Qdlin_skewness = skew(x)
    Qdlin_auc = auc(x_data, x)                 #auc：area under curve
    cycle_feat = list((Qdlin_min, Qdlin_max, Qdlin_mean, Qdlin_var, Qdlin_median, Qdlin_skewness, Qdlin_auc))
    return cycle_feat


def extract_local_fea(batch, index, start_cycle, end_cycle):

    bat_data = []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        cycle_data = []
        # extract data on cycle level
        for cycle in range(start_cycle, end_cycle + 1):
            cycle_temp=[]
            #raw
            cycle_temp.append(batch[cell_no]['summary']['cycle'][cycle])
            cycle_temp.extend(cycle_feat_extract(batch[cell_no]['cycles'][str(cycle)]['Qdlin']))
            cycle_temp.extend(cycle_feat_extract(batch[cell_no]['cycles'][str(cycle)]['Tdlin']))
            cycle_temp.extend(cycle_feat_extract(batch[cell_no]['cycles'][str(cycle)]['dQdV']))
            cycle_temp.extend(cycle_feat_extract(batch[cell_no]['cycles'][str(cycle)]['V']))
            cycle_temp.append(batch[cell_no]['summary']['IR'][cycle])
            cycle_temp.append(batch[cell_no]['summary']['QD'][cycle])
            cycle_temp.append(batch[cell_no]['summary']['QC'][cycle])
            cycle_data.append(cycle_temp)
        bat_data.append(cycle_data)
    return np.asarray(bat_data)


def main():
    model_name = 'hybrid'
    print("Start to create dataset for: ", model_name)
    # Load Data
    batch1_file = './Data/batch1_corrected.pkl'
    batch2_file = './Data/batch2_corrected.pkl'
    batch3_file = './Data/batch3_corrected.pkl'
    if os.path.exists(batch1_file) and os.path.exists(batch2_file) and os.path.exists(batch3_file):  # 判断括号内的文件是否存在
        batch1 = pickle.load(open(batch1_file, 'rb'))
        batch2 = pickle.load(open(batch2_file, 'rb'))
        batch3 = pickle.load(open(batch3_file, 'rb'))
    else:
        print("Can't find the batch data in Directory './Data' ")
        exit()

    numBat1 = len(batch1.keys())
    numBat2 = len(batch2.keys())
    numBat3 = len(batch3.keys())
    print('numBat1:', numBat1)
    print('numBat2:', numBat2)
    print('numBat3:', numBat3)
    numBat = numBat1 + numBat2 + numBat3  # 124
    print('numBat:', numBat)
    bat_dict = {**batch1, **batch2, **batch3}

    test_ind = np.hstack((np.arange(0, (numBat1 + numBat2), 2), 83))
    test_ind = np.delete(test_ind, [21])  # Remove 1 bad battery as paper
    train_ind = np.arange(1, (numBat1 + numBat2 - 1), 2)
    secondary_test_ind = np.arange(numBat - numBat3, numBat)

    _, train_y = extract_bat_cycle_life(bat_dict, train_ind)
    test_y_pri, _ = extract_bat_cycle_life(bat_dict, test_ind)
    test_y_sec, _ = extract_bat_cycle_life(bat_dict, secondary_test_ind)
    
    
    
    train_y = np.expand_dims(np.array(train_y), axis=1)
    test_y_pri = np.expand_dims(np.array(test_y_pri), axis=1)
    test_y_sec = np.expand_dims(np.array(test_y_sec), axis=1)
    
    train_x_encoder = extract_local_fea(bat_dict, train_ind, start_cycle=1, end_cycle=100)
    test_x_pri_encoder = extract_local_fea(bat_dict, test_ind, start_cycle=1, end_cycle=100)
    test_x_sec_encoder = extract_local_fea(bat_dict, secondary_test_ind, start_cycle=1, end_cycle=100)

    print("Feature Preparing for Hybird Model")
    train_x_att = extract_fea_for_hybrid_model(bat_dict, train_ind)
    test_x_pri_att = extract_fea_for_hybrid_model(bat_dict, test_ind)
    test_x_sec_att = extract_fea_for_hybrid_model(bat_dict, secondary_test_ind)

    print("train_x_encoder shape ={}, train_y shape ={}, train_x_att shape={}".format(train_x_encoder.shape, train_y.shape, train_x_att.shape))  # format：把后面3个填入3个大括号内
    print("test_x_pri shape ={}, test_y_pri shape ={}, test_x_pri_att shape={}".format(test_x_pri_encoder.shape, test_y_pri.shape, test_x_pri_att.shape))
    print("test_x_sec shape ={}, test_y_sec shape ={}, test_x_sec_att shape={}".format(test_x_sec_encoder.shape, test_y_sec.shape,test_x_sec_att.shape))

    # Max_min Normalization
    v_max = train_x_encoder.max(axis=(0, 1), keepdims=True)
    v_min = train_x_encoder.min(axis=(0, 1), keepdims=True)

    train_x_encoder_nor = (train_x_encoder - v_min) / (v_max - v_min)
    test_x_pri_encoder_nor = (test_x_pri_encoder - v_min) / (v_max - v_min)
    test_x_sec_encoder_nor = (test_x_sec_encoder - v_min) / (v_max - v_min)

    # scaler = StandardScaler()

    scaler_att = StandardScaler()
    train_x_att_nor = scaler_att.fit_transform(train_x_att)
    test_x_pri_att_nor = scaler_att.transform(test_x_pri_att)
    test_x_sec_att_nor = scaler_att.transform(test_x_sec_att)

    # No Normalization for y
    train_y_nor = train_y

    dataset = {}

    dataset['train_x_encoder'] = train_x_encoder_nor
    dataset['train_x_att'] = train_x_att_nor
    dataset['train_y'] = train_y_nor

    dataset['eva_x_encoder'] = train_x_encoder_nor
    dataset['eva_x_att'] = train_x_att_nor
    dataset['eva_y'] = train_y_nor

    dataset['test_x_pri_encoder'] = test_x_pri_encoder_nor
    dataset['test_x_pri_att'] = test_x_pri_att_nor
    dataset['test_y_pri'] = test_y_pri

    dataset['test_x_sec_encoder'] = test_x_sec_encoder_nor
    dataset['test_x_sec_att'] = test_x_sec_att_nor
    dataset['test_y_sec'] = test_y_sec

    data_path = './processed_data/first_100_cycle_data_' + model_name + '.pt'
    torch.save(dataset, data_path)


print("End")

if __name__ == '__main__':
    main()