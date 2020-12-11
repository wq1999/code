from __future__ import absolute_import
import torch
import numpy as np
import random
from tqdm import tqdm
from dataset_r import AudioDataset, VideoDataset
from torch.utils.data import DataLoader
from net_r import Audio_CNN
from net_r import Visual_CNN
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import pearsonr


def collate_fn_rnn(batch):
    imgs_batch, label_batch, label1_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    label1_batch = [torch.tensor(l) for l, imgs in zip(label1_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    labels1_tensor = torch.stack(label1_batch)
    return {'frame': imgs_tensor, 'valence': labels_tensor, 'arousal': labels1_tensor}


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

h, w = 224, 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

with open('../data_r/mediaeval2016_annotations/MEDIAEVAL16-Global_prediction.txt') as f:
    res = f.readlines()

path_test = '../data_r/input_test_audio/'

test_file = [path_test + r.split('\t')[0].split('.')[0] for r in res]
test_v_class = [float(r.split('\t')[1]) for r in res]
test_a_class = [float(r.split('\t')[2].rstrip()) for r in res]

with open('../data_r/mediaeval2016_annotations/MEDIAEVAL16-Global_prediction.txt') as f:
    res = f.readlines()

path_test = '../data_r/input_test_image/'

test_file1 = [path_test + r.split('\t')[0].split('.')[0] for r in res]
test_v_class1 = [float(r.split('\t')[1]) for r in res]
test_a_class1 = [float(r.split('\t')[2].rstrip()) for r in res]

test_ds = AudioDataset(ids=test_file, labels=test_v_class, labels1=test_a_class)
test_ds1 = VideoDataset(ids=test_file1, labels=test_v_class1, labels1=test_a_class, transform=val_transform)

audio, label, label1 = test_ds[10]
print(audio.shape, label, label1, torch.min(audio), torch.max(audio))

imgs, label, label1 = test_ds1[10]
print(imgs.shape, label, label1, torch.min(imgs), torch.max(imgs))

batch_size = 16

test_dl_a = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
test_dl_v = DataLoader(test_ds1, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_rnn,
                       drop_last=True, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_a = Audio_CNN()
model_a.to(device)

model_v = Visual_CNN()
model_v.to(device)

net_weights = torch.load('../models/audio-cnn_r_19.pth')
model_a.load_state_dict(net_weights)
model_a.eval()

net_weights1 = torch.load('../models/visual-cnn_r_13.pth')
model_v.load_state_dict(net_weights1)
model_v.eval()

lgt = []
lgt1 = []
lpred = []
lpred1 = []

for (dl, dic) in tqdm(zip(test_dl_a, test_dl_v)):
    audio = dl[0].to(device)
    label = dl[1].to(device)
    label1 = dl[2].to(device)

    image = dic['frame'].to(device)
    label2 = dic['valence'].to(device)
    label3 = dic['arousal'].to(device)

    output = model_a(audio)
    output1 = model_v(image)

    valence1 = output[0].squeeze()
    valence2 = output1[0].squeeze()

    arousal1 = output[1].squeeze()
    arousal2 = output1[1].squeeze()

    lgt.extend(label.data.cpu().numpy())
    pt = np.float32(valence1.data.cpu().numpy()) * 0.6 + np.float32(valence2.data.cpu().numpy()) * 0.4
    lpred.extend(pt)

    lgt1.extend(label1.data.cpu().numpy())
    pt = np.float32(arousal1.data.cpu().numpy()) * 0.6 + np.float32(arousal2.data.cpu().numpy()) * 0.4
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
