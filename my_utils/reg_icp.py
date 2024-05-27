# -*- codeing = utf-8 -*-
# @Time : 2024-05-23 20:49
# @Author : 张庭恺
# @File : reg_icp.py
# @Software : PyCharm
import copy

import open3d as o3d
import numpy as np

# 读取点云文件
# source = o3d.io.read_point_cloud("../data/1.ply")
# target = o3d.io.read_point_cloud("../data/lingjian1-1.ply")

# 可视化源点云和目标点云
# o3d.visualization.draw_geometries([source, target], window_name="Source and Target Point Clouds")

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# voxel_size = 10  # 自定义的体素大小
# voxel_size = 0.05  # 原始的：自定义的体素大小
# source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
# target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        # ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

# result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

# def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
def refine_registration(source, target, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point clouds.")
    print("   This time we use a strict distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, result_ransac)

# print(result_icp)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target], window_name="Registration Result")

# draw_registration_result(source, target, result_icp.transformation)
