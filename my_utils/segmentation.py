import open3d as o3d
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from .open3d_test import read_npy_to_point_cloud,read_txt_to_point_cloud
# print(sys.path)
# print(os.getcwd())


# 加载点云数据
# shape = [4809,6]

def segmentation_plan_file(file_name:str , out_dir:str = None):
    if file_name.endswith('.txt'):
        pcd = read_txt_to_point_cloud(file_name)
    elif file_name.endswith('.npy'):
        pcd = read_npy_to_point_cloud(file_name)
    else:
        pcd = o3d.io.read_point_cloud(file_name)

    # 对点云进行RANSAC平面分割
    plane_model, inliers = pcd.segment_plane(distance_threshold=1, ransac_n=3, num_iterations=1000)
    # plane_model, inliers = pcd.segment_sphere(distance_threshold=0.01, ransac_n=3, num_iterations=1)
    print('Plane model:', plane_model)
    # 返回的是一个shape = [4809]

    # TODO 所以我们这里是可以多是实现plane分割来实现与c++那边一样的分割效果的

    # 保存成各个部分的
    # 提取平面点云
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # 保存已经分割出来的面
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    o3d.io.write_point_cloud(os.path.join(out_dir, os.path.basename(file_name).split('.')[0] + '_0f.ply'), inlier_cloud,write_ascii=True)

    res_points_num = np.asarray(outlier_cloud.points).shape[0]
    save_idx = 1
    while res_points_num > 10000:
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

        inlier_cloud = outlier_cloud.select_by_index(inliers)
        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
        # 保存接下来分割出来的面
        if len(inliers) > 2000:
            o3d.io.write_point_cloud(os.path.join(out_dir, os.path.basename(file_name).split('.')[0] + f'_{save_idx}f.ply'), inlier_cloud,write_ascii=True)
        save_idx += 1
        res_points_num = np.asarray(outlier_cloud.points).shape[0]
        # 可视化分割结果

    # o3d.visualization.draw_geometries([inlier_cloud])
    # o3d.visualization.draw_geometries([outlier_cloud])
    # print('保存完成')

    # max_label = labels.max()
    # # 这是一个颜色映射表，将shape = [4809] 映射到 [4809,4]的着色表中
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

def segmentation_plan(pcd:o3d.geometry.PointCloud , out_dir:str = None):

    # 对点云进行RANSAC平面分割
    plane_model, inliers = pcd.segment_plane(distance_threshold=1, ransac_n=3, num_iterations=1000)
    # plane_model, inliers = pcd.segment_sphere(distance_threshold=0.01, ransac_n=3, num_iterations=1)
    print('Plane model:', plane_model)
    # 返回的是一个shape = [4809]

    # TODO 所以我们这里是可以多是实现plane分割来实现与c++那边一样的分割效果的

    # 保存成各个部分的
    # 提取平面点云
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # 保存已经分割出来的面
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    o3d.io.write_point_cloud(os.path.join(out_dir, '0f.ply'), inlier_cloud,write_ascii=True)

    res_points_num = np.asarray(outlier_cloud.points).shape[0]
    save_idx = 1
    while res_points_num > 10000:
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

        inlier_cloud = outlier_cloud.select_by_index(inliers)
        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
        # 保存接下来分割出来的面
        if len(inliers) > 2000:
            o3d.io.write_point_cloud(os.path.join(out_dir,f'{save_idx}f.ply'), inlier_cloud,write_ascii=True)
        save_idx += 1
        res_points_num = np.asarray(outlier_cloud.points).shape[0]
        # 可视化分割结果

    # o3d.visualization.draw_geometries([inlier_cloud])
    # o3d.visualization.draw_geometries([outlier_cloud])
    # print('保存完成')

def segmentation_cluster(file_name:str , out_dir:str):
    if file_name.endswith('.txt'):
        pcd = read_txt_to_point_cloud(file_name)
    elif file_name.endswith('.npy'):
        pcd = read_npy_to_point_cloud(file_name)
    else:
        pcd = o3d.io.read_point_cloud(file_name)
    # 密度聚类示例
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        labels = np.array(pcd.cluster_dbscan(eps=1.5, min_points=10, print_progress=True))

    # 返回的是一个shape = [4809]

    points = np.asarray(pcd.points)
    # colors = np.asarray(pcd.colors)
    # 保存成各个部分的
    classes = labels.max() + 1
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(classes):
        res_idx = labels == i
        idx = np.where(res_idx)[0]


        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(points[idx,:])
        # pcd_part.colors = o3d.utility.Vector3dVector(colors[idx,:])

        o3d.io.write_point_cloud(os.path.join(out_dir, f"part_{i}.ply"), pcd_part,write_ascii=True)
    print('保存完成')



if __name__ == '__main__':
    # file_name = '../pointcloud_data/1.ply'
    # out_dir = '../pointcloud_data/segmentaion_ply'
    # segmentation_plan(file_name,out_dir)
    # pcd = o3d.io.read_point_cloud('../pointcloud_data/1.ply')
    # data = np.asarray(pcd.points)
    # print(data)
    # pcd = o3d.geometry.PointCloud(pcd)
    pass
