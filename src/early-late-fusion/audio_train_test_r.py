from __future__ import absolute_import
import pandas as pd
import torch
import numpy as np
import random
from dataset_r import AudioDataset
from torch.utils.data import DataLoader
from net_r import Audio_CNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import pearsonr
from tqdm import tqdm
import torch.nn as nn


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


with open('../data_r/mediaeval2016_annotations/ACCEDEranking.txt') as f:
    res = f.readlines()

res = res[1:]
path_train = '../data_r/input_train_audio/'

file = [path_train + r.split('\t')[1].split('.')[0] for r in res]
v_class = [float(r.split('\t')[4]) for r in res]
a_class = [float(r.split('\t')[5].rstrip()) for r in res]

data = pd.DataFrame(index=range(len(res)))
data['file'] = file
data['v_class'] = v_class
data['a_class'] = a_class

train, valid = train_test_split(data['file'], test_size=0.1, random_state=42)
train_indx = train.index
valid_indx = valid.index

train_ids = data.iloc[train_indx, :]['file'].tolist()
train_labels_v = data.iloc[train_indx, :]['v_class'].tolist()
train_labels_a = data.iloc[train_indx, :]['a_class'].tolist()

valid_ids = data.iloc[valid_indx, :]['file'].tolist()
valid_labels_v = data.iloc[valid_indx, :]['v_class'].tolist()
valid_labels_a = data.iloc[valid_indx, :]['a_class'].tolist()

with open('../data_r/mediaeval2016_annotations/MEDIAEVAL16-Global_prediction.txt') as f:
    res = f.readlines()

path_test = '../data_r/input_test_audio/'

test_file = [path_test + r.split('\t')[0].split('.')[0] for r in res]
test_v_class = [float(r.split('\t')[1]) for r in res]
test_a_class = [float(r.split('\t')[2].rstrip()) for r in res]

train_ds = AudioDataset(ids=train_ids, labels=train_labels_v, labels1=train_labels_a)
val_ds = AudioDataset(ids=valid_ids, labels=valid_labels_v, labels1=valid_labels_a)
test_ds = AudioDataset(ids=test_file, labels=test_v_class, labels1=test_a_class)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Audio_CNN()
model.to(device)

max_epoch = 20
lr = 0.001

criterion = nn.MSELoss()

optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95)

print('Train :')
for epoch in range(0, max_epoch):

    print('Epoch:', epoch + 1)
    loss_history = []
    model.train()
    for data, target, target1 in tqdm(train_dl):

        img = data.to(device)
        target = target.to(device).float()
        target1 = target1.to(device).float()

        output = model(img)

        loss_v = criterion(output[0].squeeze(), target)

        loss_a = criterion(output[1].squeeze(), target1)

        loss = loss_v + loss_a

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(float(loss))

    scheduler.step()

    print('train_loss: {:.4}'.format(torch.mean(torch.Tensor(loss_history))))

    loss_history = []
    lgt = []
    lgt1 = []
    lpred = []
    lpred1 = []
    model.eval()
    for data, target, target1 in tqdm(val_dl):
        img = data.to(device)
        target = target.to(device).float().squeeze()
        target1 = target1.to(device).float().squeeze()

        output = model(img)

        loss_v = criterion(output[0].squeeze(), target)

        loss_a = criterion(output[1].squeeze(), target1)

        loss = loss_v + loss_a

        valence = output[0].squeeze()
        arousal = output[1].squeeze()

        lgt.extend(target.data.cpu().numpy())
        pt = np.float32(valence.data.cpu().numpy())
        lpred.extend(pt)

        lgt1.extend(target1.data.cpu().numpy())
        pt = np.float32(arousal.data.cpu().numpy())
        lpred1.extend(pt)

        loss_history.append(float(loss))

    gt = np.array(lgt).tolist()
    pred = np.squeeze(np.array(lpred)).tolist()
    mse_v = mean_squared_error(gt, pred)
    pcc_v = pearsonr(gt, pred)

    gt1 = np.array(lgt1).tolist()
    pred1 = np.squeeze(np.array(lpred1)).tolist()
    mse_a = mean_squared_error(gt1, pred1)
    pcc_a = pearsonr(gt1, pred1)

    print('valid_loss: {:.4}|valid_mse_v: {:.4}|valid_mse_a: {:.4}|valid_pcc_v: {:.4}|valid_pcc_a: {:.4}'.format(
        torch.mean(torch.Tensor(loss_history)),
        mse_v,
        mse_a,
        pcc_v[0],
        pcc_a[0],
    ))

    torch.save(model.state_dict(), '../checkpoints/audio-cnn_r_%d.pth' % (epoch + 1))


# test accuracy
print('Test Accuracy:')
for epoch in range(0, max_epoch):

    print('Epoch:', epoch)
    model = Audio_CNN()
    model.to(device)

    net_weights = torch.load('../checkpoints/audio-cnn_r_%d.pth' % (epoch + 1))
    model.load_state_dict(net_weights)
    model.eval()

    lgt = []
    lgt1 = []
    lpred = []
    lpred1 = []

    for data, label, label1 in tqdm(test_dl):
        img = data.to(device)
        target = label.to(device).float()
        target1 = label1.to(device).float()

        output = model(img)

        valence = output[0].squeeze()
        arousal = output[1].squeeze()

        lgt.extend(target.data.cpu().numpy())
        pt = np.float32(valence.data.cpu().numpy())
        lpred.extend(pt)

        lgt1.extend(target1.data.cpu().numpy())
        pt = np.float32(arousal.data.cpu().numpy())
        lpred1.extend(pt)

    gt = np.array(lgt).tolist()
    pred = np.squeeze(np.array(lpred)).tolist()
    mse_v = mean_squared_error(gt, pred)
    pcc_v = pearsonr(gt, pred)

    gt1 = np.array(lgt1).tolist()
    pred1 = np.squeeze(np.array(lpred1)).tolist()
    mse_a = mean_squared_error(gt1, pred1)
    pcc_a = pearsonr(gt1, pred1)

    print('Valence Audio Test MSE:', mse_v)
    print('Arousal Audio Test MSE:', mse_a)
    print('Valence Audio Test PCC:', pcc_v[0])
    print('Arousal Audio Test PCC:', pcc_a[0])

    del model
    torch.cuda.empty_cache()
