from __future__ import absolute_import
import torch
import numpy as np
import random
from torch import nn
from tqdm import tqdm
from load_file import load_data, load_dev_test, create_test
from dataset import AudioDataset, VideoDataset
from torch.utils.data import DataLoader
from net import Audio_CNN
from net import Visual_CNN
from torchvision import transforms


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

path_lbl = '../data/annotations'
path_split = '../data/MEDIAEVAL-mapping'
path_audio = '../data/input_aduio'
path_image = '../data/input_image/MediaEval'

path_lbl_m = path_lbl + '/MEDIAEVALaffect.txt'
path_lbl_a = path_lbl + '/ACCEDEaffect.txt'

path_split_dev = path_split + '/shots-devset-nl.txt'
path_split_test = path_split + '/shots-testset-nl.txt'

data = load_data(path_lbl_m, path_lbl_a)
dev_file, test_file = load_dev_test(path_split_dev, path_split_test)

test_ids, test_labels_v, test_labels_a = create_test(data, test_file, path_audio)
test_ids1, test_labels1_v, test_labels1_a = create_test(data, test_file, path_image)

test_ds = AudioDataset(ids=test_ids, labels=test_labels_v, labels1=test_labels_a)
test_ds1 = VideoDataset(ids=test_ids1, labels=test_labels1_v, labels1=test_labels1_a, transform=val_transform)

audio, label, label1 = test_ds[10]
print(audio.shape, label-1, label1-1, torch.min(audio), torch.max(audio))

imgs, label, label1 = test_ds1[10]
print(imgs.shape, label-1, label1-1, torch.min(imgs), torch.max(imgs))

batch_size = 16

test_dl_a = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
test_dl_v = DataLoader(test_ds1, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_rnn,
                       drop_last=True, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_a = Audio_CNN()
model_a.to(device)

model_v = Visual_CNN()
model_v.to(device)

net_weights = torch.load('../models/audio-cnn_10.pth')
model_a.load_state_dict(net_weights)
model_a.eval()

net_weights1 = torch.load('../models/visual-cnn_10.pth')
model_v.load_state_dict(net_weights1)
model_v.eval()

acc = 0.
acc1 = 0.
i = 0

for (dl, dic) in tqdm(zip(test_dl_a, test_dl_v)):

    audio = dl[0].to(device)
    label = dl[1].to(device)
    label1 = dl[2].to(device)

    image = dic['frame'].to(device)
    label2 = dic['valence'].to(device)
    label3 = dic['arousal'].to(device)

    output = model_a(audio)
    output1 = model_v(image)

    valence1 = output[0]
    valence2 = output1[0]

    arousal1 = output[1]
    arousal2 = output1[1]

    outputs1 = nn.functional.softmax(valence1, dim=1) * 0.5 + nn.functional.softmax(valence2, dim=1) * 0.5
    outputs1 = torch.argmax(outputs1, dim=1)

    acc += (outputs1 == label).float().mean().item()

    outputs2 = nn.functional.softmax(arousal1, dim=1) * 0.5 + nn.functional.softmax(arousal2, dim=1) * 0.5
    outputs2 = torch.argmax(outputs2, dim=1)

    acc1 += (outputs2 == label1).float().mean().item()

    i += 1

print('Valence Late Fusion Test Accuracy:', acc / i)
print('Arousal Late Fusion Test Accuracy:', acc1 / i)
