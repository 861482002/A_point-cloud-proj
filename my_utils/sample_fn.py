# -*- codeing = utf-8 -*-
# @Time : 2024-05-23 17:36
# @Author : 张庭恺
# @File : format.py
# @Software : PyCharm

import numpy as np
import open3d as o3d

def random_select_points(pc:np.ndarray, m) -> np.ndarray:
    if m < 0:
        idx = np.arange(pc.shape[0])
        np.random.shuffle(idx)
        return pc[idx, :]
    # pc=np.array(pc)
    # print(len(pc.points))
    # n = len(pc.points)
    n = pc.shape[0]
    print(n)
    replace = False if n >= m else True
    idx = np.random.choice(n, size=(m, ), replace=replace)
    return pc[idx, :]