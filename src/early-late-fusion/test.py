from __future__ import absolute_import
import torch
import numpy as np
import random
from torch import nn
from tqdm import tqdm
from load_file import load_data, load_dev_test, create_test, create_dev
from dataset import VideoDataset
from torch.utils.data import DataLoader
from net import Visual_CNN
import torchvision.transforms as transforms


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


# calculate accuracy
def calculat_acc(output, target):
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    return (output == target).float().mean()


path_lbl = '../data/annotations'
path_split = '../data/MEDIAEVAL-mapping'
path_img = '../data/input_image/MediaEval'

path_lbl_m = path_lbl + '/MEDIAEVALaffect.txt'
path_lbl_a = path_lbl + '/ACCEDEaffect.txt'

path_split_dev = path_split + '/shots-devset-nl.txt'
path_split_test = path_split + '/shots-testset-nl.txt'

data = load_data(path_lbl_m, path_lbl_a)
dev_file, test_file = load_dev_test(path_split_dev, path_split_test)
data_t, dev_ids, dev_v_class, dev_a_class = create_dev(data, dev_file, path_img)
test_ids, test_labels_v, test_labels_a = create_test(data, test_file, path_img)

h, w = 224, 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
test_ds = VideoDataset(ids=test_ids, labels=test_labels_v, labels1=test_labels_a, transform=val_transform)

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

net_weights = torch.load('../models/visual-cnn_10.pth')
model.load_state_dict(net_weights)
model.eval()

acc = 0.
acc1 = 0.
acc2 = 0.
i = 0

for data, label, label1 in tqdm(test_dl):
    img = data.to(device)
    label = label.to(device)
    label1 = label1.to(device)

    output = model(img)

    valence = output[0]
    arousal = output[1]

    output1 = nn.functional.softmax(valence, dim=1)
    output1 = torch.argmax(output1, dim=1)

    acc += (output1 == label).float().mean().item()

    output2 = nn.functional.softmax(arousal, dim=1)
    output2 = torch.argmax(output2, dim=1)

    acc1 += (output2 == label1).float().mean().item()

    i += 1

print('Valence Video Test Accuracy:', acc / i)
print('Arousal Video Test Accuracy:', acc1 / i)

del model
torch.cuda.empty_cache()
