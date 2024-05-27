import open3d as o3d
import os
import sys
import numpy as np
from typing import Optional, Tuple, Union, List, Dict
from pathlib import Path
from dataclasses import dataclass
from .show_3d import *
from stl import mesh


def read_txt_to_point_cloud(filename:str):
    points = []
    with open(filename,'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) == 6:  # 确保每一行都有三个坐标值
                x, y, z, r, g, b = map(float, parts)
                points.append([x, y, z, r, g, b])

    # 将读取到的点云数据转换为numpy数组，然后创建Open3D点云对象
    point_xyz_rgb = np.asarray(points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_xyz_rgb[:,:3])
    point_cloud.colors = o3d.utility.Vector3dVector(point_xyz_rgb[:,3:] / 255)
    return point_cloud

def read_npy_to_point_cloud(filename:str):
    pointcloud = np.load(filename).reshape(-1,6)


    # 将读取到的点云数据转换为numpy数组，然后创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
    point_cloud.colors = o3d.utility.Vector3dVector(pointcloud[:,3:] / 255)
    return point_cloud

def save_ply(point:Union[np.ndarray,o3d.geometry.PointCloud],out_dir:str):

    if isinstance(point,np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        o3d.io.write_point_cloud(out_dir,pcd,write_ascii=True)
    elif isinstance(point,o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(out_dir,point,write_ascii=True)


def estimate_normal(pcd:o3d.geometry.PointCloud, radius:float=0.1, max_nn:int=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return pcd


if __name__ == '__main__':


    # cloud = o3d.io.read_point_cloud('./segmentaion_ply/part_0.ply')
    # radius = 0.1  # 邻域搜索半径，根据点云密度调整
    # max_nn = 30  # 搜索的最大邻域点数量
    # cloud = estimate_normal(cloud, radius, max_nn)
    # print(cloud)

    # path = './val_data_npy/clouds1_f0.npy'
    # load = np.load('./segmentaion/part1.npy')
    # print(load.shape)

    # loadtxt = np.genfromtxt('../pointcloud_data/segmented_clouds1.txt',delimiter=' ',skip_header=1)

    # point = read_txt_to_point_cloud('conferenceRoom_1.txt')
    show_3d_file('../data/lingjian1-1.ply')
    # show_3d('./val_data/clouds1_f1.ply')
    # for file in os.listdir('val_data'):
    #     if file.startswith('clouds1'):
    #         point_cloud = o3d.io.read_point_cloud(os.path.join('val_data', file))
    #         # o3d.visualization.draw_geometries([point_cloud])
    #         print(np.asarray(point_cloud.points).shape)
    #         # o3d.visualization.draw_geometries([point])
    #         # print(point.shape)
    
    #         # 创建可视化窗口
    #         visualizer = o3d.visualization.Visualizer()
    #         visualizer.create_window()
    
    #         # 添加点云到可视化窗口
    #         visualizer.add_geometry(point_cloud)
    
    #         # 设置渲染选项
    #         render_options = visualizer.get_render_option()
    #         render_options.point_size = 3  # 设置点的大小
    
    
    #         # 显示点云
    #         visualizer.run()
    #         visualizer.destroy_window()
    # file = mesh.Mesh.from_file('./lingjian1-1.STL')
    # print(file)
    pass