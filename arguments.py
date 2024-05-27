# -*- codeing = utf-8 -*-
# @Time : 2024-05-22 11:15
# @Author : 张庭恺
# @File : arguments.py
# @Software : PyCharm
from dataclasses import dataclass
import torch
import sys
import os
import glob
import numpy as np

from PCReg.build_reg import Registration
from Pointnet_Pointnet2.build_pointnet import PointNet

@dataclass()
class Config:
	'''
	1.配准模型的参数

	2.pointnet网络参数

	3.CAD离散成点云的参数
	'''
	# 1、配准模型参数
	use_reg: bool = False
	reg_model_path: str = r'D:\A_point-cloud-proj\PCReg\pth\test_min_loss.pth'
	in_dim: int = 3
	niters: int = 8
	gn: bool = False
	device:str = 'cuda'
	infer_npts:int = 2048
	voxelsize: float = 5


	# 2、pointnet网络参数
	num_class:int = 40      #预训练模型设置的参数
	normal_channel:bool = False
	pointnet_model_path:str = r'D:\A_point-cloud-proj\Pointnet_Pointnet2\log\classification\pointnet2_cls_ssg\checkpoints\best_model.pth'

	# CAD离散成点云的参数
	sample_num:int = 90000
	pass


def test_registraion(cfg: Config) -> torch.Tensor:
	# sys.path.append(r'D:\A_point-cloud-proj\PCReg')
	registration = Registration(cfg)
	src = torch.rand(2048, 3).transpose(0,1)
	ref = torch.rand(2048, 3).transpose(0,1)
	pred_ref_cloud = registration(src, ref)
	print(pred_ref_cloud)
	return pred_ref_cloud[-1]
	pass

def test_pointnet(cfg: Config) -> torch.Tensor:
	pointnet2 = PointNet(cfg)
	x = torch.rand(1, 3, 4096)
	featurn = pointnet2(x,512)
	print(featurn.shape)
	return featurn
	pass

if __name__ == '__main__':
	cfg = Config()
	# test_pointnet(cfg)
	cfg.text = 'nihao'
	pass
