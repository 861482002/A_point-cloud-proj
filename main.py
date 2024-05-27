# -*- codeing = utf-8 -*-
# @Time : 2024-05-23 15:16
# @Author : 张庭恺
# @File : main.py
# @Software : PyCharm

import open3d as o3d
import numpy as np

import torch
import my_utils
import matplotlib.pyplot as plt

fontdict = {"family": "KaiTi", "size": 15, "color": "r"}

# 创建一些数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
z = [2, 3, 5, 7, 11]
figure = plt.figure()
subplot = figure.add_subplot(111,projection = '3d')

# 创建图表
subplot.plot(x, y,z)

# 添加标题和轴标签
subplot.set_title('3D绘图',fontdict=fontdict)
subplot.set_xlabel('X轴',fontdict=fontdict)
subplot.set_ylabel('Y轴',fontdict=fontdict)

# 在图中添加文本
subplot.text2D(0.1, 0.1, '质数', color='red',transform=subplot.transAxes,fontdict=fontdict)

# 显示图表
plt.show()


