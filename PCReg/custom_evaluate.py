import argparse
import numpy as np
import open3d as o3d
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import CustomData
from models import IterativeBenchmark,Benchmark, icp
from metrics import compute_metrics, summary_metrics, print_metrics
from utils import npy2pcd, pcd2npy


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--root', required=True, help=r'D:\PCReg.PyTorch-main\read_data\textsegment')
    parser.add_argument('--infer_npts', type=int, required=True,
                        help='2048')
    parser.add_argument('--in_dim', type=int, default=3,
                        help='3 for (x, y, z) or 6 for (x, y, z, nx, ny, nz)')
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--checkpoint', default='',
                        help='pth/test_min_loss.pth')
    parser.add_argument('--method', default='benchmark',
                        help='benchmark')
    parser.add_argument('--cuda', action='store_true',
                        help='true')
    parser.add_argument('--show', action='store_true',
                        help='whether to visualize')
    args = parser.parse_args()
    return args

def pc_denormalize(pc_normalized, mean, m):
    pc = pc_normalized * m
    pc += mean
    return pc

def evaluate_benchmark(args, test_loader):
    import mayavi.mlab as mlab

    model = IterativeBenchmark(in_dim=args.in_dim,
                               niters=args.niters,
                               gn=args.gn)
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for i, (ref_cloud, src_cloud, gtR, gtt,mean_1,m_1) in tqdm(enumerate(test_loader)):
            if i % 2 == 1:
                continue

            # print('ref_cloud',ref_cloud)
            # print('src_cloud',src_cloud)
            # print('gtR',gtR)
            # print('gtt',gtt)
            if args.cuda:
                ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                                 gtR.cuda(), gtt.cuda()
            tic = time.time()
            R, t, pred_ref_cloud = model(src_cloud.permute(0, 2, 1).contiguous(),
                    ref_cloud.permute(0, 2, 1).contiguous())
            toc = time.time()
            dura.append(toc - tic)
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

            if args.show:
                # ref_cloud = pc_denormalize(ref_cloud, mean_1, m_1)
                # print('ref_cloud',ref_cloud)
                # src_cloud = pc_denormalize(src_cloud, mean_1, m_1)
                # print('src_cloud', src_cloud)
                # pred_ref_cloud = pc_denormalize(pred_ref_cloud[-1], mean_1, m_1)
                # print('pred_ref_cloud', pred_ref_cloud)
                ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
                src_cloud = torch.squeeze(src_cloud).cpu().numpy()
                pred_ref_cloud = torch.squeeze(pred_ref_cloud[-1]).cpu().numpy()
                print('ref_cloud',ref_cloud)
                print('src_cloud',src_cloud)
                print('pred_ref_cloud',len(pred_ref_cloud))
                # pred_ref_cloud = torch.squeeze(pred_ref_cloud).cpu().numpy()
                pcd1 = npy2pcd(ref_cloud, 0)
                pcd2 = npy2pcd(src_cloud, 1)
                pcd3 = npy2pcd(pred_ref_cloud, 2)

                o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
                # points = pred_ref_cloud
                # x = points[:, 0]
                # y = points[:, 1]
                # z = points[:, 2]
                # fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 640))
                # mlab.points3d(x, y, z, mode="point", color=(0, 0, 1), figure=fig)   # 蓝色 A到B配准之后的结果
                # x = ref_cloud[:, 0]
                # y = ref_cloud[:, 1]
                # z = ref_cloud[:, 2]
                # mlab.points3d(x, y, z,x, mode="point", color=(0, 1, 0), figure=fig)   # 绿色  原CAD数据
                # x = src_cloud[:, 0]
                # y = src_cloud[:, 1]
                # z = src_cloud[:, 2]
                # mlab.points3d(x, y, z,x, mode="point", color=(1, 0, 0), figure=fig)   # 红色  目标点云数据
                # mlab.show()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


def evaluate_icp(args, test_loader):
    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for i, (ref_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
        if args.cuda:
            ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda()

        ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
        src_cloud = torch.squeeze(src_cloud).cpu().numpy()

        tic = time.time()
        R, t, pred_ref_cloud = icp(npy2pcd(src_cloud), npy2pcd(ref_cloud))
        toc = time.time()
        R = torch.from_numpy(np.expand_dims(R, 0)).to(gtR)
        t = torch.from_numpy(np.expand_dims(t, 0)).to(gtt)
        dura.append(toc - tic)

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

        if args.show:
            pcd1 = npy2pcd(ref_cloud, 0)
            pcd2 = npy2pcd(src_cloud, 1)
            pcd3 = pred_ref_cloud
            o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


if __name__ == '__main__':
    seed = 222
    random.seed(seed)
    np.random.seed(seed)

    args = config_params()

    test_set = CustomData(args.root, args.infer_npts, False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if args.method == 'benchmark':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_benchmark(args, test_loader)
        print_metrics(args.method,
                      dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    elif args.method == 'icp':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_icp(args, test_loader)
        print_metrics(args.method, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic,
                      t_isotropic)
    else:
        raise NotImplementedError