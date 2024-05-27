# -*- codeing = utf-8 -*-
# @Time : 2024-05-22 11:12
# @Author : 张庭恺
# @File : build_reg.py
# @Software : PyCharm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List,Tuple,Sequence,Optional,Union
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from .data import CustomData
from .models import IterativeBenchmark, icp
from .metrics import compute_metrics, summary_metrics, print_metrics
from .utils import npy2pcd, pcd2npy


class Registration:
	'''
	    This class is used to registration the point cloud.

	    Args: cfg: Config
	'''

	def __init__(self, cfg):
		self.args = cfg
		self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
		self.model = IterativeBenchmark(cfg.in_dim, cfg.niters, cfg.gn).to(self.device)
		self.cpk = torch.load(cfg.reg_model_path)
		self.model.load_state_dict(self.cpk)


	def __call__(self, source: torch.Tensor, ref_pcd: torch.Tensor) -> torch.Tensor:
		'''
		配准输入的没有batch维度
		Parameters
		----------
		source      shape[ n_points , 3]
		ref_pcd     shape[ n_points , 3]

		Returns
		-------

		'''


		self.model.eval()
		with torch.no_grad():
			source = source.unsqueeze(0).permute(0,2,1).to(self.device)
			ref_pcd = ref_pcd.unsqueeze(0).permute(0,2,1).to(self.device)

			# 返回张量的shape [n_points, 3]
			transformed_xs = self.model.only_reg(source, ref_pcd)

		return transformed_xs.squeeze()
