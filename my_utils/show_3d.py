# -*- codeing = utf-8 -*-
# @Time : 2024-05-23 19:47
# @Author : 张庭恺
# @File : show_3d.py
# @Software : PyCharm
import numpy as np
import open3d as o3d
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .open3d_test import *

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])


def show_ndarray(point: np.ndarray):
	point_cloud = o3d.geometry.PointCloud()
	points = o3d.utility.Vector3dVector(point)
	point_cloud.points = points


	o3d.visualization.draw_geometries([point_cloud,axis])


def show_ndarrays(points: List[np.ndarray]):
	draw_list = []
	for point in points:
		point_cloud = o3d.geometry.PointCloud()
		point = o3d.utility.Vector3dVector(point)
		point_cloud.points = point
		draw_list.append(point_cloud)
	draw_list.append(axis)
	o3d.visualization.draw_geometries(draw_list)


def show_3d_file(file_name: str):
	if file_name.endswith(('.ply', '.pcd')):
		pcd = o3d.io.read_point_cloud(file_name)

	elif file_name.endswith('.txt'):
		pcd = read_txt_to_point_cloud(file_name)
	elif file_name.endswith('.npy'):
		pcd = read_npy_to_point_cloud(file_name)
	else:
		raise ValueError("Unsupported file format")
	o3d.visualization.draw_geometries([pcd,axis], window_name=file_name)

def show_3d_files(file_names: List[str]):

	show_list = []
	for file in file_names:
		if file.endswith(('.ply', '.pcd')):
			pcd = o3d.io.read_point_cloud(file)
			show_list.append(pcd)
		elif file.endswith('.txt'):
			pcd = read_txt_to_point_cloud(file)
			show_list.append(pcd)
		elif file.endswith('.npy'):
			pcd = read_npy_to_point_cloud(file)
			show_list.append(pcd)
		else:
			raise ValueError("Unsupported file format")
	show_list.append(axis)
	o3d.visualization.draw_geometries(show_list, window_name='3D Point Cloud')


def show_3d_plt(ndarray: np.ndarray):
	figure = plt.figure()
	ax = figure.add_subplot(111, projection='3d')

	ax.scatter(ndarray[:, 0], ndarray[:, 1], ndarray[:, 2], s=1)
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')
	ax.set_title('3D Scatter Plot with Uniform Point Size')
	plt.show()


def visualize_point_clouds(pcd1:np.ndarray, pcd2:np.ndarray, data1, data2):
	fig = plt.figure(figsize=(14, 6))
	fontdict = {"family": "KaiTi", "size": 15, "color": "r"}
	# 第一个子图
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], c='r', marker='o', s=1, label='Point Cloud 1')
	ax1.set_xlabel('X',fontdict=fontdict)
	ax1.set_ylabel('Y',fontdict=fontdict)
	ax1.set_zlabel('Z',fontdict=fontdict)
	ax1.set_title('Point Cloud 1',fontdict=fontdict)
	ax1.legend()

	# 第二个子图
	ax2 = fig.add_subplot(122, projection='3d')
	ax2.scatter(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], c='b', marker='o', s=1, label='Point Cloud 2')
	ax2.set_xlabel('X',fontdict=fontdict)
	ax2.set_ylabel('Y',fontdict=fontdict)
	ax2.set_zlabel('Z',fontdict=fontdict)
	ax2.set_title('Point Cloud 2',fontdict=fontdict)
	ax2.legend()

	ax2.text2D(0.05, 0.95, f'平面度：{data1}', transform=ax2.transAxes,fontdict=fontdict)
	ax2.text2D(0.05, 1, f'曲面度：{data2}', transform=ax2.transAxes,fontdict=fontdict)

	plt.tight_layout()
	plt.show()


