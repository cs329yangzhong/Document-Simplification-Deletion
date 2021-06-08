from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
from torch import optim
import time, random
import os
from tqdm import tqdm
from model import BiLSTM_Classifier
import numpy as np
import argparse
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader
from data_loading import get_batch, get_ValidOrTest

def get_accuracy(truth, pred):
    assert (len(truth) == len(pred))
    #print((truth))
    #print((pred))
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    #truth = truth.detach().cpu().numpy()
    #pred = pred.detach().cpu().numpu()
   
    # add f1 metrics values.
    all_value = precision_recall_fscore_support(truth,pred, average="binary")
    return right / len(truth), all_value


def train_epoch_progress(model, loss_function, optimizer, args, model_type, className):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0

    train_data = os.listdir("./data/train_batch/")
    batch_total = len(train_data)//2

    time1 = time.time()
    
    for batch_id in range(batch_total):
        # Initialized model hidden for each batch.
        model.zero_grad()
        optimizer.zero_grad()
        if args.model == "BiLSTM":
            model.hidden = model.init_hidden(args)

        embed, pos,  label = get_batch(batch_id,className, delete = args.delete)
       
        truth_res += list(label)
        pred = model.forward(embed, pos)
        target = torch.Tensor(label).cuda().unsqueeze(0).reshape(-1,1)
        if args.model == "CNN":
            target = torch.Tensor(label).cuda() 
        #print(pred.shape)
        #print(target.shape) 
        loss = loss_function(pred, target)
        if args.model == "BiLSTM":
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()
        #print(nn.Sigmoid(pred.unsqueeze(1)))
        pred_res +=list(torch.round(torch.sigmoid(pred).cpu()).detach().numpy())

        avg_loss += loss.item()
        count += 1
    print("for this epoch, we use %s s"%(time.time()-time1))
    avg_loss /= (batch_total)
    acc, f1_metrics = get_accuracy(truth_res, pred_res)
    return avg_loss, acc

def train_epoch(model, train_iter, loss_function, optimizer):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc, f1_metrics  = get_accuracy(truth_res, pred_res)
    return avg_loss, acc, f1_metrics


def evaluate(model, VorT, loss_function, name, classname,model_type, args):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    all_file = get_ValidOrTest(type=VorT, class_name=classname,args=args,delete=args.delete, file_path="./data/valid_test_batch/")
    for item in all_file:
        item = list(item)
        label = item[0]
        pos = item[1]
        #print(item[2])
        sent = np.array(item[2])
        model.batch_size = len(label)
        if args.model == "BiLSTM":
            model.hidden = model.init_hidden(args)
        pred = model(sent, pos)
        truth_res += list(label)
         
        target  = torch.Tensor(label).cuda().unsqueeze(0).reshape(-1,1) 
        if (args.model == "CNN"):
            target = torch.Tensor(label).cuda()
        loss = loss_function(pred, target)
        #print(pred)
        if (args.model == "CNN"):
            pred_res += [int(y) for y in list(torch.round(torch.sigmoid(pred).cpu()).detach().numpy())]
        else:

            pred_res += [int(y[0]) for y in list(torch.round(torch.sigmoid(pred).cpu()).detach().numpy())]
        avg_loss += loss.item()
    #print(truth_res)
    #rint(pred_res)
    #raise ValueError('A very specific bad thing happened.')
    acc, f1_metrics = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f' % (avg_loss/len(all_file), acc*100))
    return avg_loss / len(all_file), acc, f1_metrics

