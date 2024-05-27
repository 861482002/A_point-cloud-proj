# -*- codeing = utf-8 -*-
# @Time : 2024-05-22 17:15
# @Author : 张庭恺
# @File : integration.py
# @Software : PyCharm
import copy

import numpy as np
import sys
import os

import open3d.cpu.pybind.visualization
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arguments import Config
from PCReg.build_reg import Registration
from Pointnet_Pointnet2.build_pointnet import PointNet
from my_utils import *

PointCloud = o3d.geometry.PointCloud
ndarray = np.ndarray


class Integration:
	'''
	集成所有部分的类
	需要做的事情
	1.构建配准模型
	2.构建pointnet++模型
	3.包含CAD离散方法
	4.包含特征匹配方法
	5.分割方法
	'''

	segmented_ply: List = []
	segmented_cad: List = []
	postseg_dir: str = None

	def __init__(self, cfg: Config):
		self.cfg = cfg
		self.reg_model = Registration(cfg) if cfg.use_reg else None

		self.pointnet_model = PointNet(cfg)

		self.postreg_ply = None

		pass

	def reg_ply_cad_deepmodel(self, source_ply: str, target_cad: str, cfg: Config) -> np.ndarray:
		'''
		配准点云和CAD
		Parameters
		----------
		source_ply:是一个.ply后缀的文件
		target_cad:是一个.stl后缀的文件

		Returns
		-------
		postreg_ply: 配准后的点云 shape[infer_npts,3]
		'''

		# 1.读取ply文件 和cad文件
		pcd_ply = read_ply(source_ply)
		pcd_cad = read_stl(target_cad, sample_num=cfg.sample_num)

		# 统一两个点云的点的数量
		# 返回的数据shape: [infer_npts, 3]
		ply_point = np.asarray(pcd_ply.points)
		cad_point = np.asarray(pcd_cad.points)
		ply_point = random_select_points(ply_point, cfg.infer_npts)
		cad_point = random_select_points(cad_point, cfg.infer_npts)

		# show_ndarrays([ply_point, cad_point])
		# 将读取到的数据转换成torch.Tensor
		# shape:[infer_npts, 3]
		ply_point = torch.tensor(ply_point)
		cad_point = torch.tensor(cad_point)
		# 模型输入的数据都是[batch,c,n_points]
		# 返回的是一个tensor
		self.postreg_ply = self.reg_model(ply_point, cad_point).cpu().numpy()
		# show_ndarrays([self.postreg_ply,cad_point])
		return self.postreg_ply

	def reg_ply_cad_icp(self, source: PointCloud, target: PointCloud, voxelsize: float = 5) -> Tuple[
		PointCloud, ndarray]:
		'''
		使用ICP方法配准两个模型
		Parameters
		----------
		source  :待配准的点云
		target  :目标点云
		voxelsize   :下采样的时候体素的大小

		Returns
		-------

		'''
		# 首先进行下采样、法向量估计、快速点特征直方图(FPFH)
		ply_down, ply_fpfh = preprocess_point_cloud(source, voxelsize)
		cad_down, cad_fpfh = preprocess_point_cloud(target, voxelsize)

		# 进行ICP配准
		# 使用的方法是RANSAC 随机一致性采样法
		result_ransac = execute_global_registration(ply_down, cad_down, ply_fpfh, cad_fpfh, voxel_size=voxelsize)

		# 精配准
		result_icp = refine_registration(ply_down, cad_down, voxelsize, result_ransac)

		# 配准后的点云
		# 这个方法是在原地修改
		post_reg_ply = copy.deepcopy(source)
		post_reg_ply = post_reg_ply.transform(result_icp.transformation)

		return post_reg_ply, result_icp.transformation

	def pointnet_forward(self, points_list: List[np.ndarray], return_dim=512) -> np.ndarray:
		'''
		批量pointnet++的特征提取
		Parameters
		----------
		batch_points:List [ndarray:[n_points , 3]]

		Returns
		-------
		features: shape [batch, 512/256]
		'''
		features = []
		for cur_point in points_list:
			single_point = torch.tensor(cur_point, dtype=torch.float32).unsqueeze(0)
			single_featurn = self.pointnet_model(single_point, return_dim=return_dim)
			single_featurn = single_featurn.cpu().numpy()
			features.append(single_featurn)
		features = np.concatenate(features, axis=0)
		return features

		pass

	def segment_postreg_ply(self, ply_file: Optional[str] = None, out_dir: Optional[str] = None) -> List:
		# TODO 先写死，后续调用c++的程序进行分割
		segmented_ply_dir = './data/segmentply'
		segmented_ply_dir_list = os.listdir(segmented_ply_dir)
		self.postseg_ply_dir = out_dir
		for ply in segmented_ply_dir_list:
			cur_ply_path = os.path.join(segmented_ply_dir, ply)
			cur_pcd = read_ply(cur_ply_path)
			points = np.asarray(cur_pcd.points)
			self.segmented_ply.append(points)

		# shape不一样
		return self.segmented_ply

	def segment_cad(self, cad_file: Optional[str] = None, out_dir: Optional[str] = None) -> List:
		# TODO 先写死，后续调用c++的程序进行分割

		segmented_cad_dir = './data/segmentply'
		segmented_cad_dir_list = os.listdir(segmented_cad_dir)
		self.postseg_cad_dir = out_dir
		for cad in segmented_cad_dir_list:
			cur_cad_path = os.path.join(segmented_cad_dir, cad)
			cur_pcd = read_ply(cur_cad_path)
			points = np.asarray(cur_pcd.points)
			self.segmented_cad.append(points)
		# shape不一样
		return self.segmented_cad
		pass

	def select_plane(self,plane_list: List[ndarray]) -> np.ndarray:
		# TODO 先写死，后续会添加参数作为判断条件
		idx = 0
		pcd = plane_list[idx]
		return pcd

		pass

	def faiss_match(self):
		# TODO 特征检索匹配
		pass

	def visualize(self, pcd1: ndarray, pcd2: ndarray, *args):
		# TODO 可视化
		visualize_point_clouds(pcd1, pcd2, *args)

		pass


if __name__ == '__main__':
	# 初始化
	cfg = Config()
	integration = Integration(cfg)
	'''测试配准                     pass'''
	# source = read_ply('./data/1.ply')
	# target = read_ply('./data/lingjian1-1.ply')
	# postreg_source = integration.reg_ply_cad_icp(source, target)

	# postreg_ply = integration.reg_ply_cad('./data/1.ply', './data/lingjian1-1.STL', cfg)
	# print(postreg_ply.shape)
	'''测试文件类型转换             pass'''
	# cad2ply('./data/lingjian2-1.STL','./data/lingjian2-1.ply',90000)
	# cad2ply('./data/lingjian3-1.STL','./data/lingjian3-1.ply',90000)
	'''测试pointnet++             pass'''
	# cfg = Config()
	# integration = Integration(cfg)
	# x = torch.rand(1, 4096, 3)
	# y = integration.pointnet_forward(x,512)
	# print(y.shape)
	'''测试可视化                    pass'''
	# point_cloud = read_ply('./data/my_segment/segmented_cloud0.pcd')
	# point_cloud = np.asarray(point_cloud.points)
	# show_plt(point_cloud)

	'''整体流程                         pass'''

	# 1、配准
	source = read_ply('./data/1.ply')
	target = read_ply('./data/lingjian1-1.ply')
	postreg_source, transformation = integration.reg_ply_cad_icp(source, target)
	# 2、分割 配准后的点云和原始cad
	postreg_source_dir = './temp/seg_postreg_source'
	segmentation_plan(postreg_source, postreg_source_dir)
	target_dir = './temp/seg_target'
	segmentation_plan(target, target_dir)
	# 3、读取分割后的数据
	seg_postreg_source = []
	seg_target = []

	for file in os.listdir(postreg_source_dir):
		pcd = read_ply(os.path.join(postreg_source_dir, file))
		seg_postreg_source.append(np.asarray(pcd.points))

	for file in os.listdir(target_dir):
		pcd = read_ply(os.path.join(target_dir, file))
		seg_target.append(np.asarray(pcd.points))
	# 4、通过给定的条件选取一个面
	selected_cad_plane = integration.select_plane(seg_target)

	# 5、使用pointnet++来提取特征
	# 返回的shape[分割后的平面数量，256]
	feature_ply = integration.pointnet_forward(seg_postreg_source, return_dim=256)
	# 返回的shape[1,256]
	feature_plane = integration.pointnet_forward([selected_cad_plane], return_dim=256)

	matched_file = feature_match(feature_ply, postreg_source_dir, feature_plane)

	matched_file = os.path.join(postreg_source_dir, matched_file)
	print(matched_file)
	matched_ply = read_ply(matched_file)
	matched_ply.paint_uniform_color([1.0, 0, 0])
	selected_cad_plane = o3d.utility.Vector3dVector(selected_cad_plane)
	selected_pcd = o3d.geometry.PointCloud(selected_cad_plane)
	selected_pcd.paint_uniform_color([0, 1.0, 0])

	o3d.visualization.draw_geometries([matched_ply, selected_pcd,axis])
	#
	# matched_ply = np.asarray(matched_ply.points)
	#
	#
	# # 6、可视化
	# visualize_point_clouds(matched_ply, selected_cad_plane, 0, 0)
	pass
