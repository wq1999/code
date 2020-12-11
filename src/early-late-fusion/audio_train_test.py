from __future__ import absolute_import
import torch
import numpy as np
import random
from torch import nn
from tqdm import tqdm
from load_file import load_data, load_dev_test, creat_train_val, create_test, create_dev
from dataset import AudioDataset
from torch.utils.data import DataLoader
from net import Audio_CNN


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


# calculate accuracy
def calculat_acc(output, target):
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    return (output == target).float().mean()


path_lbl = '../data/annotations'
path_split = '../data/MEDIAEVAL-mapping'
path_img = '../data/input_aduio'

path_lbl_m = path_lbl + '/MEDIAEVALaffect.txt'
path_lbl_a = path_lbl + '/ACCEDEaffect.txt'

path_split_dev = path_split + '/shots-devset-nl.txt'
path_split_test = path_split + '/shots-testset-nl.txt'

data = load_data(path_lbl_m, path_lbl_a)
dev_file, test_file = load_dev_test(path_split_dev, path_split_test)
data_t, dev_ids, dev_v_class, dev_a_class = create_dev(data, dev_file, path_img)
train_ids, train_labels_v, train_labels_a, valid_ids, valid_labels_v, valid_labels_a = creat_train_val(data_t)
test_ids, test_labels_v, test_labels_a = create_test(data, test_file, path_img)

train_ds = AudioDataset(ids=train_ids, labels=train_labels_v, labels1=train_labels_a)
val_ds = AudioDataset(ids=valid_ids, labels=valid_labels_v, labels1=valid_labels_a)
test_ds = AudioDataset(ids=test_ids, labels=test_labels_v, labels1=test_labels_a)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Audio_CNN()
model.to(device)

# valence class weights
w1, w2, w3 = 1. / np.log(data_t['v_class'].value_counts())
weights_v = [w1, w2, w3]

# arousal class weights
w11, w22, w33 = 1. / np.log(data_t['a_class'].value_counts())
weights_a = [w11, w22, w33]

# frame-level train

max_epoch = 10
best_acc_v = 0.
best_acc_a = 0.
valid_accs_v = []
valid_accs_a = []
valid_losses = []
lowest_val_loss = np.inf
lr = 0.001

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
    for data, target, target1 in tqdm(train_dl):

        img = data.to(device)
        target = target.to(device)
        target1 = target1.to(device)

        output = model(img)

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
    for data, target, target1 in tqdm(val_dl):
        img = data.to(device)
        target = target.to(device)
        target1 = target1.to(device)

        output = model(img)

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

    torch.save(model.state_dict(), '../checkpoints/audio-cnn_%d.pth' % (epoch + 1))

# test accuracy
print('Test Accuracy:')
for epoch in range(0, max_epoch):

    print('Epoch:', epoch)
    model = Audio_CNN()
    model.to(device)

    net_weights = torch.load('../checkpoints/audio-cnn_%d.pth' % (epoch + 1))
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

    print('Valence Audio Test Accuracy:', acc / i)
    print('Arousal Audio Test Accuracy:', acc1 / i)

    del model
    torch.cuda.empty_cache()
