import os
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image


class AudioDataset(Dataset):
    def __init__(self, ids, labels, labels1,):

        self.ids = ids
        self.labels = labels
        self.labels1 = labels1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        dir = self.ids[idx]
        path2imgs = os.listdir(dir)

        path2imgs = [os.path.join(dir, i) for i in path2imgs]

        label = self.labels[idx] + 1
        label1 = self.labels1[idx] + 1

        p2i = path2imgs[0]
        frame = np.load(p2i)
        frame = torch.from_numpy(frame)

        return frame, label, label1


class VideoDataset(Dataset):
    def __init__(self, ids, labels, labels1, transform):
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.labels1 = labels1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        dir = self.ids[idx]
        path2imgs = os.listdir(dir)

        path2imgs.sort(key=lambda x: int(x[5:-4]))
        path2imgs = [os.path.join(dir, i) for i in path2imgs]

        label = self.labels[idx] + 1
        label1 = self.labels1[idx] + 1

        frames = []
        for p2i in path2imgs:
            frame = Image.open(p2i)
            frames.append(frame)

        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr) > 0:
            frames_tr = torch.stack(frames_tr)

        return frames_tr, label, label1
