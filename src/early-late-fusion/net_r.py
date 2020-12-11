from torch.nn.utils.weight_norm import weight_norm
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import models


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


# concat fusion
class Classifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(Classifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Audio_CNN(nn.Module):
    def __init__(self, drop_prob=0.3, num_layers=3, num_classes=1):
        super(Audio_CNN, self).__init__()

        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.num_classes = num_classes

        # context gating
        self.gate_fe = ContextGating(128)

        # intermediate_fc
        self.intermediate_fc = nn.Sequential(
            nn.Linear(128 * 2, 128),
            ContextGating(128))

        # lstm
        self.LSTM = nn.LSTM(input_size=128, hidden_size=512, num_layers=self.num_layers, batch_first=True)
        self.Dropout = nn.Dropout(p=self.drop_prob)
        self.w = nn.Parameter(torch.ones(8), requires_grad=True)

        # fc
        self.down_fc = nn.Sequential(
            nn.Linear(128+512, 512),
            ContextGating(512))

        self.fc_valence = nn.Sequential(
            Classifier(512, 256, num_classes, 0.3))

        self.fc_arousal = nn.Sequential(
            Classifier(512, 256, num_classes, 0.3))

    def forward(self, x_3d):

        cnn_embedding_out = []

        for t in range(x_3d.size(1)):
            x = x_3d[:, t, :]

            # context gating
            x = self.gate_fe(x)
            cnn_embedding_out.append(x)

        frame_out = torch.stack(cnn_embedding_out, dim=0)
        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        lstm_f = cnn_embedding_out

        # lstm
        lstm_out, _ = self.LSTM(lstm_f)
        vecs = []
        seqlen = lstm_out.size(1)
        for s in range(seqlen):
            vec = self.Dropout(lstm_out[:, s, :]) * self.w[s]
            vecs.append(vec)

        lstm_out = torch.stack(vecs, dim=1)
        lstm_out = torch.mean(lstm_out, dim=1)

        # video-level mean and std
        x1 = torch.mean(frame_out, 0)
        tmp = frame_out.detach().cpu().numpy()
        x2 = np.std(tmp, axis=0)
        x2 = np.nan_to_num(x2)
        x2 = torch.from_numpy(x2).cuda()
        v_fea = torch.cat([x1, x2], dim=1)
        v_fea = F.normalize(v_fea, p=2, dim=1)
        v_fea = self.intermediate_fc(v_fea)

        # fusion
        out = torch.cat([lstm_out, v_fea], dim=1)
        out = F.normalize(out, p=2, dim=1)

        # classifier
        out = self.down_fc(out)
        valence = self.fc_valence(out)
        arousal = self.fc_arousal(out)

        return valence, arousal


class Visual_CNN(nn.Module):
    def __init__(self, drop_prob=0.3, pretrained=True, num_layers=3, num_classes=3):
        super(Visual_CNN, self).__init__()

        self.pretrained = pretrained
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.num_classes = num_classes

        # cnn resnet
        pretrained_cnn = models.resnet101(pretrained=self.pretrained)
        cnn_layers = list(pretrained_cnn.children())[:-1]
        self.cnn = nn.Sequential(*cnn_layers)

        # context gating
        self.gate = ContextGating(2048)

        # cnn as feature extractor
        for param in self.cnn.parameters():
            param.requires_grad = False

        # intermediate_fc
        self.intermediate_fc = nn.Sequential(
            nn.Linear(2048 * 2, 2048),
            ContextGating(2048))

        # lstm
        self.LSTM = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=self.num_layers, batch_first=True)
        self.Dropout = nn.Dropout(p=self.drop_prob)
        self.w = nn.Parameter(torch.ones(8), requires_grad=True)

        # fc
        self.down_fc = nn.Sequential(
            nn.Linear(2048 + 1024, 2048),
            ContextGating(2048))

        self.fc_valence = nn.Sequential(
            nn.Linear(2048, 1))

        self.fc_arousal = nn.Sequential(
            nn.Linear(2048, 1))

    def forward(self, x_3d):

        cnn_embedding_out = []

        for t in range(x_3d.size(1)):
            x = self.cnn(x_3d[:, t, :, :, :])
            x = torch.flatten(x, start_dim=1)

            # context gating
            x = self.gate(x)
            cnn_embedding_out.append(x)

        frame_out = torch.stack(cnn_embedding_out, dim=0)
        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        lstm_f = cnn_embedding_out

        # lstm
        lstm_out, _ = self.LSTM(lstm_f)
        vecs = []
        seqlen = lstm_out.size(1)
        for s in range(seqlen):
            vec = self.Dropout(lstm_out[:, s, :]) * self.w[s]
            vecs.append(vec)

        lstm_out = torch.stack(vecs, dim=1)
        lstm_out = torch.mean(lstm_out, dim=1)

        # video-level mean and std
        x1 = torch.mean(frame_out, 0)
        tmp = frame_out.detach().cpu().numpy()
        x2 = np.std(tmp, axis=0)
        x2 = np.nan_to_num(x2)
        x2 = torch.from_numpy(x2).cuda()
        v_fea = torch.cat([x1, x2], dim=1)
        v_fea = F.normalize(v_fea, p=2, dim=1)
        v_fea = self.intermediate_fc(v_fea)

        # fusion
        out = torch.cat([lstm_out, v_fea], dim=1)
        out = F.normalize(out, p=2, dim=1)
        out = self.down_fc(out)

        # classifier
        valence = self.fc_valence(out)
        arousal = self.fc_arousal(out)

        return valence, arousal
