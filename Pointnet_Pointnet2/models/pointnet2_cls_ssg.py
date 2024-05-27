import torch
import torch.nn as nn
import torch.nn.functional as F
from Pointnet_Pointnet2.models.pointnet2_utils import PointNetSetAbstraction
import numpy as np
# from .pointnet2_utils import PointNetSetAbstraction
import os
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    # def forward(self, xyz,save_path,file_name):
    def feature_extraction(self, xyz:torch.Tensor,return_dim:int,require_norm:bool = False) ->torch.Tensor:

        # xyz.shape: [ batch, 3, num_points]
        # return_dim: 256 or 512
        # norm : 返回的特征是否需要标准化处理
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        x = self.fc1(x)  # [1,512]
        if return_dim == 512:
            if require_norm:
                return self.bn1(x)
            else:
                return x

        x = self.bn1(x)
        x = self.drop1(F.relu(x))

        x = self.fc2(x)

        if return_dim == 256:
            if require_norm:
                return self.bn2(x)
            else:
                return x


    def forward(self, xyz:torch.Tensor):
        # xyz.shape: [ batch, 3, num_points]
        # return_dim: 256 or 512
        # norm : 返回的特征是否需要标准化处理
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # print('123123123',l3_points.shape)
        #
        # final_output = l3_points.cpu().numpy().squeeze()
        # print(final_output.shape)
        # final_output = final_output.reshape((1,1024))
        # np.savetxt('array_data11_2.txt', final_output)
        # print(final_output.shape)
        # print(final_output)
        x = l3_points.view(B, 1024)
        x = self.fc1(x)                         #[1,512]

        final_output1 = x.cpu().numpy().squeeze()       #[512]
        # print(final_output.shape)
        # final_output = final_output.reshape((1,512))
        # np.savetxt(os.path.join(save_path,f'{file_name}.txt'), final_output1)
        # print(final_output.shape)
        # print(final_output)
        print('1231231231231', x.shape)


        print(x.shape)

        x = self.bn1(x)
        x = self.drop1(F.relu(x))

        x = self.fc2(x)

        x = self.drop2(F.relu(self.bn2(x)))
        final_output2 = x.cpu().numpy().squeeze()
        # print(final_output.shape)
        # final_output = final_output.reshape((1,512))
        # np.savetxt('array_data11_256_2.txt', final_output2)
        # print(final_output.shape)
        # print(final_output)
        print('1231231231231',x.shape)
        x = self.fc3(x)
        print('66666666666',x.shape)
        x = F.log_softmax(x, -1)


        return x, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
