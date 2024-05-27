# -*- codeing = utf-8 -*-
# @Time : 2024-05-22 17:58
# @Author : 张庭恺
# @File : io_ply.py
# @Software : PyCharm
import os

import open3d as o3d
import numpy as np
import trimesh

def read_ply(ply_path: str) -> o3d.geometry.PointCloud:
	pcd = o3d.io.read_point_cloud(ply_path)
	return pcd

def read_stl(cad_path:str,sample_num) -> o3d.geometry.PointCloud:
	# 读取CAD模型
	mesh = trimesh.load(cad_path)
	# 使用顶点和面生成均匀点云
	# 返回一个ndarray的矩阵[90000,3]
	points, _ = trimesh.sample.sample_surface(mesh, sample_num)  # 采样90000个点

	point_cloud = o3d.geometry.PointCloud()
	point_cloud.points = o3d.utility.Vector3dVector(points)

	return point_cloud

def read_txt1(filename: str):
	# 预处理所有数值都在一行内的点云，同一个点的[x,y,z]用 ','分开，不同的点之间用 ';'分开
	with open(filename, 'r') as fp:
		content = fp.read()
	points = content.split(';')
	points = [p.split(',') for p in points if len(p) != 0]
	points = np.array(points, dtype=np.float32)
	return points

def read_txt2(filename: str):
	# 使用numpy格式存储的点云文件，使用' '(空格)分割，一行一个点
	points = np.loadtxt(filename, delimiter=' ')
	return points

def read_txt_to_point_cloud(filename: str):
	# 这个暂时没用
	points = []
	with open(filename, 'r') as f:
		for line in f:
			parts = line.strip().split(' ')
			if len(parts) == 3:  # 确保每一行都有三个坐标值
				x, y, z = map(float, parts)
				points.append([x, y, z])

	# 将读取到的点云数据转换为numpy数组，然后创建Open3D点云对象
	point_xyz_rgb = np.asarray(points)
	point_cloud = o3d.geometry.PointCloud()
	point_cloud.points = o3d.utility.Vector3dVector(point_xyz_rgb)
	# point_cloud.colors = o3d.utility.Vector3dVector(point_xyz_rgb[:, 3:] / 255)
	return point_cloud