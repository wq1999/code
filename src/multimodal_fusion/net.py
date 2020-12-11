import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.nn.utils.weight_norm import weight_norm


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


class FCNet(nn.Module):
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


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
    def __init__(self, drop_prob=0.3, num_layers=3, num_classes=3):
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
            Classifier(512, 256, 3, 0.3),
            ContextGating(3))

        self.fc_arousal = nn.Sequential(
            Classifier(512, 256, 3, 0.3),
            ContextGating(3))

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

        return out, valence, arousal


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
            Classifier(2048, 1024, 3, 0.3),
            ContextGating(3))

        self.fc_arousal = nn.Sequential(
            Classifier(2048, 1024, 3, 0.3),
            ContextGating(3))

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

        return out, valence, arousal


class LateFusion_fc(nn.Module):
    """
    Use each modality's prediction as feature vector and classify it
    """

    def __init__(self, num_classes):
        super(LateFusion_fc, self).__init__()
        self.visual = Visual_CNN()
        self.audio = Audio_CNN()
        self.final_pred = nn.Linear(num_classes * 2, num_classes)

    def forward(self, frames, audio):

        _, vis, vis1 = self.visual(frames)
        _, audio, audio1 = self.audio(audio)

        pred = self.final_pred(torch.cat([vis, audio], dim=-1))
        pred1 = self.final_pred(torch.cat([vis1, audio1], dim=-1))

        return pred, pred1


class LateFusion_w(nn.Module):
    """
    weight average on each modality's prediction
    """

    def __init__(self, num_modal):
        super(LateFusion_w, self).__init__()
        self.visual = Visual_CNN()
        self.audio = Audio_CNN()
        self.w_v = nn.Parameter(torch.ones(num_modal), requires_grad=True)
        self.w_a = nn.Parameter(torch.ones(num_modal), requires_grad=True)

    def forward(self, frames, audio):
        _, vis, vis1 = self.visual(frames)
        _, audio, audio1 = self.audio(audio)

        pred = vis * self.w_v[0] + audio * self.w_v[1]
        pred1 = vis1 * self.w_a[0] + audio1 * self.w_a[1]

        return pred, pred1


class GMU(nn.Module):
    """
    Gated Multimodal Unit
    """
    def __init__(self, drpt=0.4, num_classes=3):
        super(GMU, self).__init__()

        self.visual = Visual_CNN()
        self.audio = Audio_CNN()
        self.drpt = drpt
        self.visual_redu = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(p=self.drpt))
        self.audio_redu = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(p=self.drpt))
        self.ponderation = nn.Sequential(nn.Linear(512 + 2048, 1), nn.Sigmoid())

        self.final_pred_v = nn.Linear(128, num_classes)
        self.final_pred_a = nn.Linear(128, num_classes)

    def forward(self, frame, audio):

        visual, _, _ = self.visual(frame)
        audio, _, _ = self.audio(audio)

        z = self.ponderation(torch.cat([visual, audio], 1))
        visual = self.visual_redu(visual)
        audio = self.audio_redu(audio)

        h = z * visual + (1.0 - z) * audio
        pred1 = self.final_pred_v(h)
        pred2 = self.final_pred_a(h)

        return pred1, pred2


class Attention(nn.Module):
    def __init__(self, g_dim, l_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([g_dim + l_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, g, l):
        """
        g: [batch, g_dim]
        l: [batch, t_step, f_dim]
        """
        logits = self.logits(g, l)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, g, l):
        num_t = l.size(1)
        g = g.unsqueeze(1).repeat(1, num_t, 1)
        gl = torch.cat((g, l), 2)
        joint_repr = self.nonlinear(gl)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, g_dim, l_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.g_proj = FCNet([g_dim, num_hid])
        self.l_proj = FCNet([l_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, g, l):
        """
        g: [batch, g_dim]
        l: [batch, t_step, f_dim]
        """
        logits = self.logits(g, l)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, g, l):
        batch, k, _ = l.size()
        g_proj = self.g_proj(g).unsqueeze(1).repeat(1, k, 1)
        l_proj = self.l_proj(l)
        joint_repr = l_proj * g_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_audio, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_audio
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_audio = nn.Linear(dim_out, dim_audio)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual, audio):
        squeeze_array = []
        for tensor in [visual, audio]:
            squeeze_array.append(torch.mean(tensor, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        au_out = self.fc_audio(excitation)

        vis_out = self.sigmoid(vis_out)
        au_out = self.sigmoid(au_out)

        return visual * vis_out, audio * au_out
