from __future__ import absolute_import
import torch
import numpy as np
import random
from tqdm import tqdm
from dataset_r import VideoDataset
from torch.utils.data import DataLoader
from net_r import Visual_CNN
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import pearsonr


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


def collate_fn_rnn(batch):
    imgs_batch, label_batch, label1_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    label1_batch = [torch.tensor(l) for l, imgs in zip(label1_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    labels1_tensor = torch.stack(label1_batch)
    return imgs_tensor, labels_tensor, labels1_tensor


with open('../data_r/mediaeval2016_annotations/MEDIAEVAL16-Global_prediction.txt') as f:
    res = f.readlines()

path_test = '../data_r/input_test_image/'

test_file = [path_test + r.split('\t')[0].split('.')[0] for r in res]
test_v_class = [float(r.split('\t')[1]) for r in res]
test_a_class = [float(r.split('\t')[2].rstrip()) for r in res]

h, w = 224, 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

test_ds = VideoDataset(ids=test_file, labels=test_v_class, labels1=test_a_class, transform=val_transform)

batch_size = 16

test_dl = DataLoader(test_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
                     drop_last=True, collate_fn=collate_fn_rnn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Visual_CNN()
model.to(device)

# test accuracy
print('Test Accuracy:')

model = Visual_CNN()
model.to(device)

net_weights = torch.load('../models/visual-cnn_r_13.pth')
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

print('Valence Video Test MSE:', mse_v)
print('Arousal Video Test MSE:', mse_a)
print('Valence Video Test PCC:', pcc_v[0])
print('Arousal Video Test PCC:', pcc_a[0])
