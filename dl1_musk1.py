# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:40:11 2019

@author: Amina
This code performs one 10- fold cross-validation run for MUSK-1 dataset
By default it runs for Single layer architecture. 
misvmio.py, the file used for Bags I/O has been taken from https://github.com/garydoranjr/misvm 
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

import numpy as np
from misvmio import *
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score as auc_roc
from sklearn import metrics

class MyDataset(Dataset):
    def __init__(self, bags):
        self.bags = bags

        
        
    def __getitem__(self, index):
        examples = self.bags[index]

        return examples

        
    def __len__(self):
        return len(self.bags)
            



class Net(nn.Module):
    def __init__(self,d):
        super(Net, self).__init__()
        self.out = nn.Linear(d,1)        


    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
'''
One Hidden Layer Architecture
'''
#
#class Net(nn.Module):
#    def __init__(self,d):
#        super(Net, self).__init__()
#        self.hidden1 = nn.Linear(d,d)        
#        self.out = nn.Linear(d,1)
#
#    def forward(self,x):
#        x = x.view(x.size(0), -1)
#        x = self.hidden1(x)
#        x = F.tanh(x)
#        
#        x = self.out(x)
#        return x


def create_bags(dataset='musk1'):
    data_set= parse_c45(dataset, rootdir='musk')
    bagset = bag_set(data_set)
    bags_1 = [np.array(b.to_float())[:, 2:-1] for b in bagset]
    labels = np.array([b.label for b in bagset], dtype=float)
    no_of_bags=len(bags_1)
    for b_r in range(no_of_bags):
        for b_c in range(len(bags_1[b_r])):
            bags_1[b_r][b_c]=bags_1[b_r][b_c]/np.linalg.norm(bags_1[b_r][b_c])
    labels =list( 2 * labels - 1)
    return bags_1, labels
    

aucs=[]
accs=[]
bags, labels=create_bags()
bags=np.array(bags)
labels=np.array(labels)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(bags, labels):

    bags_tr=bags[train]
    y_tr=labels[train]
    bags_ts=bags[test]
    y_ts=labels[test]
    pos_bags=bags_tr[y_tr>0]
    neg_bags=bags_tr[y_tr<0]


    
    pos=MyDataset(pos_bags)
    neg=MyDataset(neg_bags)
    
    loader_pos = DataLoader(pos, batch_size=1)
    loader_neg = DataLoader(neg, batch_size=1)
    epochs=20
    mlp=Net(166)
    mlp.cuda()
    optimizer = optim.RMSprop(mlp.parameters())
    
    all_losses=[]
    for e in range(epochs):
        l=0.0
        for idx_p, pbag in enumerate(loader_pos):
            pbag=pbag.float()
            pbag=Variable(pbag).type(torch.cuda.FloatTensor)
            p_scores=mlp.forward(pbag[0])
            max_p=torch.max(p_scores)
    
            for idx_n, nbag in enumerate(loader_neg):
                nbag=nbag.float()
                nbag=Variable(nbag).type(torch.cuda.FloatTensor)
                n_scores=mlp.forward(nbag[0])

                max_n=torch.max(n_scores)
                z=np.array([0.0])
                loss=torch.max(Variable(torch.from_numpy(z)).type(torch.cuda.FloatTensor), (max_n-max_p+1))
                l=l+float(loss)
    
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                optimizer.step()

        all_losses.append(l)
    #testing

    test=MyDataset(bags_ts)
    loader_ts=DataLoader(test, batch_size=1)
    predictions=[]

    for param in mlp.parameters():
        param.requires_grad =False
    for idx_ts, tsbag in enumerate(loader_ts):
        tsbag=tsbag.float()
        tsbag=Variable(tsbag).type(torch.cuda.FloatTensor)
        scores=mlp.forward(tsbag[0])

        predictions.append(float(torch.max(scores)))
    auc=auc_roc(y_ts, predictions)
    aucs.append(auc)
    print ('AUC=',auc)
    
    
    #accuracy
    
    f, t, a=metrics.roc_curve(y_ts, predictions)
    AN=sum(x<0 for x in y_ts)
    AP=sum(x>0 for x in y_ts)
    TN=(1.0-f)*AN
    TP=t*AP
    Acc2=(TP+TN)/len(y_ts)
    acc=max(Acc2)

    print ('accuracy=',acc )
    accs.append(acc)
    


print ("\n\nmean auc=", np.mean(aucs))
print ("mean accuracy=", np.mean(accs))
print ("Epochs=", epochs)
