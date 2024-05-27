# -*- codeing = utf-8 -*-
# @Time : 2024-05-22 14:49
# @Author : 张庭恺
# @File : xxx2xxx.py
# @Software : PyCharm


import open3d as o3d
import trimesh
import numpy as np
from .my_io import *


def cad2ply(cad_path: str, ply_path: str, sample_num: int):
	# 读取CAD模型
	mesh = trimesh.load(cad_path)
	# 使用顶点和面生成均匀点云
	points, _ = trimesh.sample.sample_surface(mesh, sample_num)  # 采样90000个点

	# points = mesh.vertices

	point_cloud = o3d.geometry.PointCloud()
	point_cloud.points = o3d.utility.Vector3dVector(points)

	# 保存为 .ply 文件
	o3d.io.write_point_cloud(ply_path, point_cloud, write_ascii=True)

def txt2ply_1(txt_path: str, ply_path: str):
	'''
	这里读取的txt文件是没有规则的txt，里面的所有数据都在一行，不同点用；分割，相同点的不同xyz用，分割
	Parameters
	----------
	txt_path
	ply_path
	'''
	points = read_txt1(txt_path)
	point_cloud = o3d.geometry.PointCloud()
	point_cloud.points = o3d.utility.Vector3dVector(points)
	o3d.io.write_point_cloud(ply_path, point_cloud, write_ascii=True)



def tex2ply2(filename: str, ply_path: str):
	points = read_txt2(filename)
	point_cloud = o3d.geometry.PointCloud()
	point_cloud.points = o3d.utility.Vector3dVector(points)
	o3d.io.write_point_cloud(ply_path, point_cloud, write_ascii=True)





if __name__ == '__main__':
	pass
