import os
import torch
import torch.nn.functional as F
import numpy as np
from numpy.random import randint


model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

data_path = r'D:\Study\mediaeval2016_test_audio'
sub_dirs = [x[0] for x in os.walk(data_path)]
sub_dirs = sub_dirs[1:]

for sub_dir in sub_dirs:

    store_path = sub_dir.replace(r'D:\Study\mediaeval2016_test_audio', r'D:\Study\input_test_audio')
    print(store_path)
    if os.path.exists(store_path):
        print(store_path)
        continue
    os.makedirs(store_path, exist_ok=True)

    files_list = os.listdir(sub_dir)
    path = os.path.join(sub_dir, files_list[0])
    vec = model.forward(path).detach().numpy()
    if vec.shape[0] > 8:
        vec = vec[:8, :]
    else:
        diff = 8 - vec.shape[0]
        for i in range(diff):
            idx = randint(vec.shape[0], size=1)
            vec = np.row_stack((vec, vec[idx, :]))

    # mean = torch.mean(vec, 0)
    # std = torch.std(vec, 0)
    # vec = torch.cat([mean, std])
    # vec = F.normalize(vec, p=2, dim=0).detach().numpy()
    print(vec.shape)
    dst = os.path.join(store_path, "feature" + ".npy")
    np.save(dst, vec)
