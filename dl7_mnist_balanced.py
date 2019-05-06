# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:59:42 2019

@author: Amina Asif
This code performs one classification run of 9 vs non 9 bags in MNIST

Implementation of class MnistBags has been taken from https://github.com/AMLab-Amsterdam/AttentionDeepMIL to 
generate balanced datasets fo MNIST MIL experiments to perform comparison with Attention networks (Ilse et al.) 

Number of training bags can be varied using num_bag parameter on line 237
"""




import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score as auc_roc


import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from torchvision import datasets, transforms

from copy import deepcopy



class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=1, num_bag=1000, seed=7, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.seed = seed
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._form_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._form_bags()

    def _form_bags(self):
        if self.train:
            train_loader = data_utils.DataLoader(datasets.MNIST('datasets\\',
                                                                train=True,
                                                                download=True,
                                                                transform=transforms.Compose([
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,), (0.3081,))])),
                                                 batch_size=self.num_in_train,
                                                 shuffle=False)

            bags_list = []
            labels_list = []
            valid_bags_counter = 0
            label_of_last_bag = 0

            for batch_data in train_loader:
                numbers = batch_data[0]
                labels = batch_data[1]

            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
                labels_in_bag = labels[indices]

                if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_train, 1))
                        label_temp = labels[index]
                        if label_temp.numpy()[0] != self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1

                    index_list = np.array(index_list)
                    labels_in_bag = labels[index_list]
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        else:
            test_loader = data_utils.DataLoader(datasets.MNIST('datasets\\',
                                                               train=False,
                                                               download=True,
                                                               transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))])),
                                                batch_size=self.num_in_test,
                                                shuffle=False)

            bags_list = []
            labels_list = []
            valid_bags_counter = 0
            label_of_last_bag = 0

            for batch_data in test_loader:
                numbers = batch_data[0]
                labels = batch_data[1]

            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))
                labels_in_bag = labels[indices]

                if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_test, 1))
                        label_temp = labels[index]
                        if label_temp.numpy()[0] != self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1

                    index_list = np.array(index_list)
                    labels_in_bag = labels[index_list]
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label
    
    
    

class AttNet(nn.Module):
    def __init__(self):
        super(AttNet, self).__init__()
        
        
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(500,1),    
        )
        
         
        
        


    def forward(self,x):
        x = x.squeeze(0)
        x=self.feature_extractor_part1(x)

        x=x.view(-1, 50*4*4)

        x=self.feature_extractor_part2(x)

        x=self.classifier(x)
        

        return x
        




        
        
################ Create bags##########################

batch_size = 1

train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                               mean_bag_length=10,
                                               var_bag_length=2,
                                               num_bag=50,                                               
                                               train=True),
                                     batch_size=1,
                                     shuffle=True)

test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                              mean_bag_length=10,
                                              var_bag_length=2,
                                              num_bag=1000,                                              
                                              train=False),
                                    batch_size=1,
                                    shuffle=False)

val_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                               mean_bag_length=10,
                                               var_bag_length=2,
                                               num_bag=100,                                               
                                               train=False),
                                     batch_size=1,
                                     shuffle=False)


pos_bags=[]
neg_bags=[]
for batch_idx, (data, label) in enumerate(train_loader):
    if float(label[0])>0:
        pos_bags.append(data)
    else:
        neg_bags.append(data)
#1/0



epochs=10 #3-4 epochs enough for training set size>=400 bags   
cnn=AttNet()  
cnn.cuda()    

#lr=0.01 for bags<=50
#lr=0.005 for bags=100
#lr=0.001 for bags>100
optimizer=optim.Adam(cnn.parameters(), lr=0.01, weight_decay=0.0001, betas=(0.9, 0.999))#lr=0.005 for other bag sizes#, betas=(0.9, 0.999), weight_decay=10e-5)
all_losses=[]

best_cnn=None 
best_loss=None 
val_auc=None
best_auc=None
for e in range(epochs):
    print ("epoch:", e)
    
    count=0.0
    for idx_p, pbag in enumerate(pos_bags):
        pbag=pbag.float()
        pbag=Variable(pbag).type(torch.cuda.FloatTensor)
        p_scores=cnn.forward(pbag)
        
        max_p=torch.max(p_scores)
        l=0.0
        loss=0.0
        for idx_n, nbag in enumerate(neg_bags):
            nbag=nbag.float()
            nbag=Variable(nbag).type(torch.cuda.FloatTensor)
            n_scores=cnn.forward(nbag[0])

            max_n=torch.max(n_scores)
            z=np.array([0.0])
            loss+=torch.max(Variable(torch.from_numpy(z)).type(torch.cuda.FloatTensor), (max_n-max_p+1))
#            loss=torch.max(torch.tensor(0.0), (max_n-max_p+1))
            l=l+float(loss)
            count+=1
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer.step()
        all_losses.append(float(loss))
            
    y_val=[]
    val_pred=[]
    for batch_idx, (data, label) in enumerate(val_loader):
        y_val.append(float(float(label[0])>0))
        valbag=data.float()
        valbag=Variable(valbag).type(torch.cuda.FloatTensor)
        scores=cnn.forward(valbag)
    
        val_pred.append(float(torch.max(scores)))
    val_auc=auc_roc(y_val, val_pred)
    if e==0:
        best_auc=val_auc
        best_cnn=deepcopy(cnn)
    if val_auc>best_auc:
        best_auc=val_auc
        best_cnn=deepcopy(cnn)

    avg_loss=l/count

            
        
#    print ("current loss=",loss)
    print ("best validation auc yet=", best_auc)

    
    
for param in best_cnn.parameters():
    param.requires_grad =False
predictions=[]
y_ts=[]
for batch_idx, (data, label) in enumerate(test_loader):
    y_ts.append(float(float(label[0])>0))
    tsbag=data.float()
    tsbag=Variable(tsbag).type(torch.cuda.FloatTensor)
    scores=best_cnn.forward(tsbag)

    predictions.append(float(torch.max(scores)))
auc=auc_roc(y_ts, predictions)

print ('Best CNN AUC=',auc)


for param in cnn.parameters():
    param.requires_grad =False
predictions=[]
y_ts=[]
for batch_idx, (data, label) in enumerate(test_loader):
    y_ts.append(float(float(label[0])>0))
    tsbag=data.float()
    tsbag=Variable(tsbag).type(torch.cuda.FloatTensor)
    scores=cnn.forward(tsbag)

    predictions.append(float(torch.max(scores)))
auc=auc_roc(y_ts, predictions)

print ('Final CNN AUC=',auc)

#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(all_losses)

#%%
#ANALYSIS

#predictions=[]
#y_ts=[]
#instance_labels=[]
#instance_scores=[]
#ts_bags_all=[]
#for batch_idx, (data, label) in enumerate(test_loader):
##    import pdb; pdb.set_trace()
#    y_ts.append(float(float(label[0])>0))
#    instance_labels.append(np.array(label[1]))
#    tsbag=data.float()
#    ts_bags_all.append(tsbag)
#    tsbag=Variable(tsbag).type(torch.cuda.FloatTensor)
#    scores=best_cnn.forward(tsbag)
#
#    predictions.append(float(torch.max(scores)))
#    instance_scores.append(np.array(scores).flatten())
#auc=auc_roc(y_ts, predictions)
##aucs.append(auc)
#print ('Best CNN AUC=',auc)
#
#
#n_p=[]
#for i in instance_labels:
#    n_p.append(np.sum(i))
#ind=np.argmax(n_p)
##ind=101
#b=np.array(ts_bags_all[ind][0])
#imgs=[]
#for b1 in b:
#    imgs.append(b1[0]) 
#for i in range(len(imgs)):
#    plt.figure();plt.imshow(imgs[i]); plt.title('score='+str(instance_scores[ind][i]))
#
#
