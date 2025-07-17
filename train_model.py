import pandas as pd
import os
import time
import numpy as np
import scipy.sparse as sp
import math
import pandas as pd
import torch
import torch.nn as nn # 构建网络模块
import torch.nn.functional as F  # 网络中函数 例如 F.relu
from torch.nn.parameter import Parameter # 构建的网络的参数
from torch.nn.modules.module import Module # 自己构建的网络需要继承的模块
import torch.optim as optim # 优化器模块
from sklearn.model_selection import StratifiedKFold
import random

import pandas as pd
import numpy as np
from collections import defaultdict
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

def cal_auc(output,labels):
    output_test=output
    output_test=output_test.cpu()
    output_test=output_test.detach().numpy()
    output_test=np.exp(output_test)
    output_test=output_test[:,1]
    labels_test=labels.cpu().numpy()
    AUROC = roc_auc_score(labels_test, output_test)
    
    precision_, recall_, _thresholds = precision_recall_curve(labels_test, output_test)

    AUPRC = auc(recall_, precision_)
    return AUROC,AUPRC

def normalize_adj(adj): # 返回的是一个sparse矩阵 需要使用todense() 转化维dense矩阵
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(
            np.int64))  # # 获得稀疏矩阵坐标 (2708, 1433)  --> (49216, 2)
    values = torch.from_numpy(sparse_mx.data)  # 相应位置的值 (49216, ) 即矩阵中的所有非零值
    shape = torch.Size(sparse_mx.shape)  # 稀疏矩阵的大小
    return torch.sparse.FloatTensor(indices, values, shape)

def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(1/DE))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H,invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G
    
def _generate_G_from_H_weight(H, W):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    n_edge = H.shape[1]
    # the weight of the hyperedge
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(1/DE))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    G = DV2 * H * W * invDE * HT * DV2
    return G

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class HGNN_res(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN_res, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_ch,n_hid)
        self.hgc1 = HGNN_conv(n_hid, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_hid)
        self.outLayer = nn.Linear(n_hid, n_class)
    def forward(self, x, G):
        x1 = F.relu(self.fc(x))
        x1= F.dropout(x1, self.dropout, training=self.training)
        #print(x.shape)
        
        x2 = F.relu(self.hgc1(x1, G)+x1) 
        x2 = F.dropout(x2, self.dropout, training=self.training) 
        
        x3 = F.relu(self.hgc2(x2, G)+x2) 
        x3 = F.dropout(x3, self.dropout, training=self.training) 
        
        x4 = F.relu(self.hgc3(x3, G)+x3) 
        x4 = F.dropout(x4, self.dropout, training=self.training) 
        x5 = self.outLayer(x4) 
        return F.log_softmax(x5, dim=1)
import random
def get_data(t,compareAllGene,bgGene,disease):
    
    candidate_gene_frame = pd.read_csv(r'./AD_candidateGene.txt',header=None)
    candidate_gene = list(candidate_gene_frame[0].values)
    print(len(candidate_gene))

    positive_gene = pd.read_csv(r'./AD_gene.csv',header=None)
    positive_gene=list(positive_gene[0].values)
    positive_gene=list(set(compareAllGene)&set(positive_gene))
    positive_gene.sort()

    res_gene = list(set(compareAllGene)-set(candidate_gene)-set(positive_gene))
    print(len(res_gene))
    random.seed(t)
    negative_gene=random.sample(res_gene,1000)       
    negative_gene.sort()
    #print(len(negative_gene))

    PPI_labels=pd.DataFrame(data=[0]*len(bgGene),index=bgGene)
    PPI_labels.loc[positive_gene,:]=1
    positive_index=np.where(PPI_labels==1)[0]
    PPI_labels.loc[negative_gene,:]=-1
    negative_index=np.where(PPI_labels==-1)[0]

    PPI_labels=pd.DataFrame(data=[0]*len(bgGene),index=bgGene)
    PPI_labels.loc[positive_gene,:]=1

    positive_index=list(positive_index)
    negative_index=list(negative_index)

    train=positive_index+negative_index
    train=np.array(train)

    the_label=pd.DataFrame(data=[1]*len(positive_index)+[0]*len(negative_index))
    the_label=the_label.values.ravel()
    return  train,the_label,PPI_labels
Genes = pd.read_csv(r'./allGene.csv',header=None)
Genes = Genes[0].values
Genes = list(Genes)

ids = ['h','c1','c2','c3','c4','c5','c6','c7','c8']
pathwayMatrix = pd.DataFrame()
gsvaFeature = pd.DataFrame()
for idd in ids:
    
    pathwayMatrixSP = sp.load_npz('./'+idd+'_GenesetsMatrix.npz')
    pathwayMatrixC = pd.DataFrame(data = pathwayMatrixSP.A,index= Genes)
    pathwayMatrix = pd.concat([pathwayMatrix,pathwayMatrixC],axis=1)
pathwayMatrix.columns = np.arange(pathwayMatrix.shape[1])


disease = 'AD'

epochs = 200
lr = 5e-3
weight_decay = 5e-6
p=0
auc_hgnn_list=list()
prc_hgnn_list=list()
for i in range(5):
    train,the_label,PPI_labels=get_data(i,Genes,Genes,disease)
    sk_X=train.reshape([-1,1])
    sfolder=StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
    for train_index,test_index in sfolder.split(sk_X,the_label):
        X_train,X_test, y_train, y_test=train[train_index],train[test_index],the_label[train_index],the_label[test_index]
        idx_train=X_train
        temp=PPI_labels.iloc[X_train]
        trainPositiveGene=list(temp.where(temp==1).dropna().index)
        positive_Matrix=pathwayMatrix.loc[trainPositiveGene].sum()
        usedColums=np.where(positive_Matrix>=3)[0]
        used_feature=pathwayMatrix.iloc[:,usedColums]
        weight = positive_Matrix[usedColums].values
        usedColSum = pathwayMatrix.iloc[:,usedColums].values.sum(0)
        weight = weight/usedColSum
        H = np.array(used_feature)
        DV = np.sum(H * weight, axis=1)
        H=H.astype('float')
        for i in range(DV.shape[0]):
            if(DV[i]==0):
                t=random.randint(0,H.shape[1]-1)
                H[i][t]=0.0001
        G=_generate_G_from_H_weight(H,weight)
        N=H.shape[0]
        adj_hyperGraph = torch.Tensor(G)
        features= torch.eye(N)
        # 构建模型
        features=features.float()
        adj_hyperGraph=adj_hyperGraph.float()
        #adj_PPI=adj_PPI.float()
        labels=torch.from_numpy(PPI_labels.values.reshape(-1,))
        model = HGNN_res(in_ch=N,n_hid=256,dropout=0)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        schedular = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,200,300,400],gamma=0.5)
        if torch.cuda.is_available():
            model.cuda()
            features = features.cuda()
            adj_hyperGraph = adj_hyperGraph.cuda()
            labels = labels.cuda()
        t_total = time.time()
        loss_values = []
        for epoch in range(epochs):
            t = time.time()
            model.train() 
            optimizer.zero_grad()
            output = model(features, adj_hyperGraph)
            loss = F.nll_loss(
                output[idx_train], labels[idx_train]
            )  # ；计算损失与准确率；交叉熵loss， 因为模型计算包含 log， 这里使用 nll_loss（CrossEntropyLoss =Softmax+Log+NLLLoss）

            loss.backward()
            optimizer.step() 
            schedular.step()
        model.eval()
        with torch.no_grad():
            output = model(features, adj_hyperGraph)
            #output=output.exp()
            AUROC_val,AUPRC_val=cal_auc(output[X_test], labels[X_test])
            print('hgnn_AUROC: {:.4f}'.format(AUROC_val.item()),
                   'hgnn_AUPRC: {:.4f}'.format(AUPRC_val.item()))
            auc_hgnn_list.append(AUROC_val.item())
            prc_hgnn_list.append(AUPRC_val.item())
print("hgnn_AUROC : {:04f}".format(np.mean(auc_hgnn_list)),
      "hgnn_AUPRC : {:04f}".format(np.mean(prc_hgnn_list)))

