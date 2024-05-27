import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils import readpcd
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform


class CustomData(Dataset):
    def __init__(self, root, npts, train=True):
        super(CustomData, self).__init__()
        dirname = 'train_data' if train else 'val_data'
        path = os.path.join(root, dirname)
        self.train = train
        self.files = [os.path.join(path, item) for item in sorted(os.listdir(path))]
        self.npts = npts

    def __getitem__(self, item):
        file = self.files[item]
        ref_cloud = readpcd(file, rtype='ply')
        # print("1",type(ref_cloud))

        ref_cloud = random_select_points(ref_cloud, m=self.npts)
        # print("2",type(ref_cloud))
        # print("11111111111111111111111", ref_cloud)
        ref_cloud,mean_1,m_1 = pc_normalize(ref_cloud)
        # mean是x y z各轴的均值，m是标准差
        R, t = generate_random_rotation_matrix(-20, 20), \
               generate_random_tranlation_vector(-0.5, 0.5)
        src_cloud = transform(ref_cloud, R, t)                  #源点云是经过目标点云旋转变化得到的
        if self.train:
            ref_cloud = jitter_point_cloud(ref_cloud)
            src_cloud = jitter_point_cloud(src_cloud)
        return ref_cloud, src_cloud, R, t,mean_1,m_1

    def __len__(self):
        return len(self.files)