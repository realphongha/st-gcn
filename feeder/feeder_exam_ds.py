# sys
import os
from re import L
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# operation
from . import tools


class FeederExamDs(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in Exam dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        # self.img_path = os.path.join(data_path, "data")
        self.lbl_path = os.path.join(data_path, "label")
        self.pose_lbl_path = os.path.join(data_path, "pose_label")
        self.labels_file = os.path.join(data_path, "labels.txt")
        # self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.ignore_empty_sample = ignore_empty_sample
        self.samples = list()
        self.labels_i = dict()
        self.labels = list()

        self.load_data()
        
    def load_data(self):
        self.sample_name = os.listdir(self.pose_lbl_path)
        
        if self.debug:
            self.sample_name = self.sample_name[:2]
            
        with open(self.labels_file, "r") as labels_file:
            lbls = labels_file.read().splitlines()
            for lbl in lbls:
                lbl = lbl.strip().split()
                if len(lbl) != 2:
                    continue
                self.labels_i[lbl[0]] = int(lbl[1])
            
        for name in self.sample_name:
            label_fn = os.path.join(self.lbl_path, name + ".txt")
            with open(label_fn, "r") as label_file:
                lbls = label_file.read().splitlines()
                for lbl in lbls:
                    lbl = lbl.strip().split()
                    if len(lbl) != 3:
                        continue
                    lbl_i, start, end = self.labels_i[lbl[0]], int(lbl[1]), int(lbl[2])
                    if end - start < 50:
                        continue
                    self.samples.append({
                        "path": os.path.join(self.pose_lbl_path, name),
                        "start": start,
                        "end": end,
                    })
                    self.labels.append(lbl_i) 
                    
        # output data shape (N, C, T, V, M)
        self.N = len(self.samples)  #sample
        self.C = 3  #channel
        self.T = 10  #frame
        self.V = 13  #joint  

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # data_numpy = np.zeros((self.C, self.T, self.V, 1))
        data = self.samples[index]
        label = self.labels[index]
        path, start, end = data["path"], data["start"], data["end"]
        data_numpy = np.zeros((self.C, end-start+1, self.V, 1))
        for i in range(start, end+1):
            fn = os.path.join(path, "%i.txt" % i)
            with open(fn, "r") as pose_file:
                pose = pose_file.read().splitlines()[:13]
                pose = [list(map(float, line.strip().split())) for line in pose]
                pose = np.array(pose)
            data_numpy[0, i-start, :, :] = np.expand_dims(pose[:, 0], 1)
            data_numpy[1, i-start, :, :] = np.expand_dims(pose[:, 1], 1)
            data_numpy[2, i-start, :, :] = np.expand_dims(pose[:, 2], 1)

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # sort by score
        for t, s in enumerate(np.zeros((data_numpy.shape[1], 1)).astype(int)):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                       0))

        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.labels >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.labels)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.labels >= 0))
        return tools.top_k_by_category(self.labels, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.labels >= 0))
        return tools.calculate_recall_precision(self.labels, score)


if __name__ == "__main__":
    ds = FeederExamDs(r"/content/exam_action_ds/train",
                      window_size=50, random_choose=True, random_move=True,)
    data, lbl = ds[0]
    print(data.shape)
    print(lbl)
    