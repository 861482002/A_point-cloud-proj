# -*- codeing = utf-8 -*-
# @Time : 2024-05-22 14:47
# @Author : 张庭恺
# @File : build_pointnet.py
# @Software : PyCharm
import os
import sys

# sys.path.insert(0, os.path.dirname(__file__))
# print(sys.path)
# from arguments import Config
from typing import List, Tuple, Sequence, Optional, Union, ClassVar
import torch
import numpy
from .models import get_model

class PointNet:
	'''
	对分割后的各个面进行特征提取，之后在通过提取到的特征进行特征检索
	num_class:int = 40      #预训练模型设置的参数
	normal_channel:bool = False
	pointnet_model_path:str
	'''

	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
		self.model = get_model(cfg.num_class, cfg.normal_channel).to(self.device)
		self.cpk = torch.load(cfg.pointnet_model_path)
		self.model.load_state_dict(self.cpk['model_state_dict'])

	def __call__(self, points: torch.Tensor, return_dim: int,norm:bool = False) -> torch.Tensor:
		# points:[batch , n_points, 3]
		# 加一个判断，使用256，还是512维度的特征作为后续检索的输入

		# 改变输入数据的shape
		self.model.eval()
		points = points.permute(0,2,1).to(self.device)
		with torch.no_grad():
			feature = self.model.feature_extraction(points, return_dim)

		# 返回的特征张量的shape : []
		return feature

if __name__ == '__main__':
	pointnet2 = PointNet()
	x = torch.rand(1,3,4096)

	pass
