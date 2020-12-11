from __future__ import absolute_import
import torch
import numpy as np
import random
from torch import nn
from tqdm import tqdm
from load_file import load_data, load_dev_test, creat_train_val, create_test, create_dev
from dataset import VideoDataset, AudioDataset
from torch.utils.data import DataLoader
from net import LateFusion_fc, LateFusion_w, GMU
from torchvision import transforms

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
path_audio = '../data/input_aduio'

path_lbl_m = path_lbl + '/MEDIAEVALaffect.txt'
path_lbl_a = path_lbl + '/ACCEDEaffect.txt'

path_split_dev = path_split + '/shots-devset-nl.txt'
path_split_test = path_split + '/shots-testset-nl.txt'

data = load_data(path_lbl_m, path_lbl_a)
dev_file, test_file = load_dev_test(path_split_dev, path_split_test)

data_t, dev_ids, dev_v_class, dev_a_class = create_dev(data, dev_file, path_img)
train_ids, train_labels_v, train_labels_a, valid_ids, valid_labels_v, valid_labels_a = creat_train_val(data_t)

data_t1, dev_ids1, dev_v_class1, dev_a_class1 = create_dev(data, dev_file, path_audio)
train_ids1, train_labels_v1, train_labels_a1, valid_ids1, valid_labels_v1, valid_labels_a1 = creat_train_val(data_t1)

h, w = 224, 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

val_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

train_ds = VideoDataset(ids=train_ids, labels=train_labels_v, labels1=train_labels_a, transform=train_transform)
val_ds = VideoDataset(ids=valid_ids, labels=valid_labels_v, labels1=valid_labels_a, transform=val_transform)

train_ds1 = AudioDataset(ids=train_ids1, labels=train_labels_v1, labels1=train_labels_a1)
val_ds1 = AudioDataset(ids=valid_ids1, labels=valid_labels_v1, labels1=valid_labels_a1)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True, collate_fn=collate_fn_rnn)
val_dl = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True, collate_fn=collate_fn_rnn)

train_dl1 = DataLoader(train_ds1, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
val_dl1 = DataLoader(val_ds1, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GMU(0.4, 3)
model.to(device)

# valence class weights
w1, w2, w3 = 1. / np.log(data_t['v_class'].value_counts())
weights_v = [w1, w2, w3]

# arousal class weights
w11, w22, w33 = 1. / np.log(data_t['a_class'].value_counts())
weights_a = [w11, w22, w33]

# train

max_epoch = 10
best_acc_v = 0.
best_acc_a = 0.
valid_accs_v = []
valid_accs_a = []
valid_losses = []
lowest_val_loss = np.inf
lr = 1e-3

class_weights_v = torch.FloatTensor(weights_v).to(device)
class_weights_a = torch.FloatTensor(weights_a).to(device)

criterion_v = nn.CrossEntropyLoss(weight=class_weights_v)
criterion_a = nn.CrossEntropyLoss(weight=class_weights_a)

optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95)

print('Train :')
for epoch in range(0, max_epoch):

    print('Epoch:', epoch + 1)
    loss_history = []
    model.train()
    for (dl, dl1) in tqdm(zip(train_dl, train_dl1)):

        img = dl[0].to(device)
        target = dl[1].to(device)
        target1 = dl[2].to(device)

        audio = dl1[0].to(device)

        output = model(img, audio)

        loss_v = criterion_v(output[0], target)

        loss_a = criterion_a(output[1], target1)

        loss = loss_v + loss_a

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(float(loss))

    scheduler.step()

    print('train_loss: {:.4}'.format(torch.mean(torch.Tensor(loss_history))))

    loss_history = []
    acc_history_v = []
    acc_history_a = []
    model.eval()
    for (dl, dl1) in tqdm(zip(val_dl, val_dl1)):

        img = dl[0].to(device)
        target = dl[1].to(device)
        target1 = dl[2].to(device)

        audio = dl1[0].to(device)

        output = model(img, audio)

        loss_v = criterion_v(output[0], target)

        loss_a = criterion_a(output[1], target1)

        loss = loss_v + loss_a

        valence = output[0]
        arousal = output[1]

        acc_v = calculat_acc(valence, target)
        acc_history_v.append(float(acc_v))

        acc_a = calculat_acc(arousal, target1)
        acc_history_a.append(float(acc_a))

        loss_history.append(float(loss))

    print('valid_loss: {:.4}|valid_acc_v: {:.4}|valid_acc_a: {:.4}'.format(
        torch.mean(torch.Tensor(loss_history)),
        torch.mean(torch.Tensor(acc_history_v)),
        torch.mean(torch.Tensor(acc_history_a)),
    ))

    valid_acc_v = torch.mean(torch.Tensor(acc_history_v))
    valid_acc_a = torch.mean(torch.Tensor(acc_history_a))
    valid_loss = torch.mean(torch.Tensor(loss_history))

    valid_accs_v.append(valid_acc_v)
    valid_accs_a.append(valid_acc_a)
    valid_losses.append(valid_loss)

    torch.save(model.state_dict(), '../checkpoints/late_fusion_%d.pth' % (epoch + 1))
