import pandas as pd
import numpy as np


def load_data(path_lbl_m, path_lbl_a):
    with open(path_lbl_m, 'r') as f:
        res = f.readlines()

    with open(path_lbl_a, 'r') as f:
        res1 = f.readlines()

    res = res + res1[1:]
    file = [r.split('\t')[1] for r in res]
    v_class = [r.split('\t')[2] for r in res]
    a_class = [r.split('\t')[3].rstrip() for r in res]

    data = pd.DataFrame(index=range(len(res[1:])))
    data['file'] = file[1:]
    data['v_class'] = v_class[1:]
    data['a_class'] = a_class[1:]
    return data


def load_dev_test(path_split_dev, path_split_test):
    with open(path_split_dev, 'r') as f:
        dev = f.readlines()

    dev_file = [d.split('\t')[1].rstrip() for d in dev]

    with open(path_split_test, 'r') as f:
        test = f.readlines()

    test_file = [t.split('\t')[1].rstrip() for t in test]
    return dev_file, test_file


def create_dev(data, dev_file, path_img):

    dev_v_class = [int(data[data['file'] == d.split('.')[0]]['v_class'].tolist()[0]) for d in dev_file]
    dev_a_class = [int(data[data['file'] == d.split('.')[0]]['a_class'].tolist()[0]) for d in dev_file]
    dev_ids = [path_img + '/' + d.split('.')[0] for d in dev_file]
    data_t = pd.DataFrame(index=range(len(dev_file)))
    data_t['file'] = dev_ids
    data_t['v_class'] = dev_v_class
    data_t['a_class'] = dev_a_class
    return data_t, dev_ids, dev_v_class, dev_a_class


def creat_train_val(data_t):
    train_indx = np.load('../data/splits/train.npy')
    valid_indx = np.load('../data/splits/valid.npy')

    train_ids = data_t.iloc[train_indx, :]['file'].tolist()
    train_labels_v = data_t.iloc[train_indx, :]['v_class'].tolist()
    train_labels_a = data_t.iloc[train_indx, :]['a_class'].tolist()

    valid_ids = data_t.iloc[valid_indx, :]['file'].tolist()
    valid_labels_v = data_t.iloc[valid_indx, :]['v_class'].tolist()
    valid_labels_a = data_t.iloc[valid_indx, :]['a_class'].tolist()

    return train_ids, train_labels_v, train_labels_a, valid_ids, valid_labels_v, valid_labels_a


def create_test(data, test_file, path_img):
    test_v_class = [int(data[data['file'] == t.split('.')[0]]['v_class'].tolist()[0]) for t in test_file]
    test_a_class = [int(data[data['file'] == t.split('.')[0]]['a_class'].tolist()[0]) for t in test_file]

    test_ids = [path_img + '/' + t.split('.')[0] for t in test_file]
    test_labels_v = test_v_class
    test_labels_a = test_a_class
    return test_ids, test_labels_v, test_labels_a
