# -*- codeing = utf-8 -*-
# @Time : 2024-05-26 17:06
# @Author : 张庭恺
# @File : module_test.py
# @Software : PyCharm
import os

from my_utils import *


def add_axis(file1, file2):
	cad = read_ply(file1)
	ply = read_ply(file2)
	'''
	X轴是红色的.
	Y轴是绿色的.
	Z轴是蓝色的.
	'''
	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

	print(f'的最小值:{np.asarray(cad.points).min(axis=0)}')
	print(f'的最大值:{np.asarray(cad.points).max(axis=0)}')
	print(f'的最小值:{np.asarray(ply.points).min(axis=0)}')
	print(f'的最大值:{np.asarray(ply.points).max(axis=0)}')

	show_3d_file('./data/lingjian1-1.ply')
	show_3d_file('./data/2.ply')
	show_3d_file('./data/3.ply')



if __name__ == '__main__':
	# my_seg_path = './data/segmentply'
	# cad_files = os.listdir(my_seg_path)
	# for file in cad_files:
	# 	if file.startswith('clouds1'):
	# 		o3d.visualization.draw_geometries([read_ply(os.path.join(my_seg_path, file))])

	# segmentation_plan('./data/1.ply','./data/my_segment_RANSAC')

	# show_3d_files(['./data/lingjian1-1.ply','./data/1.ply'])
	show_3d_file('./temp/seg_postreg_source/1f.ply')
	# files = ['./data/segmentply/clouds1_f0.ply','./data/my_segment_RANSAC/1_0f.ply']
	# show_3d_files(files)



	pass
