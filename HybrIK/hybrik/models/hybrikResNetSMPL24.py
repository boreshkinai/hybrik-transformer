# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
from typing import Tuple
from collections import namedtuple

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import SPPE
from .layers.Resnet import ResNet
from .layers.smpl.SMPL import SMPL_layer

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from .layers.smpl.lbs import rotmat_to_quat


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features))
        self.b_2 = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualBlock(torch.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int):
        super(ResidualBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width

        self.fc_layers = [torch.nn.Linear(size_in, layer_width)]
        self.relu_layers = [torch.nn.LeakyReLU(inplace=True)]
        self.fc_layers += [torch.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.relu_layers += [torch.nn.LeakyReLU(inplace=True) for _ in range(num_layers - 1)]

        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        self.relu_layers = torch.nn.ModuleList(self.relu_layers)
        self.layer_norm = LayerNorm(num_features=layer_width)
        self.dp = torch.nn.Dropout(p=dropout)
        self.shortcut = torch.nn.Linear(size_in, layer_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        h = torch.nn.functional.relu(x)
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        return self.layer_norm(self.dp(h) + shortcut)
    


ModelOutput = namedtuple(
    typename='ModelOutput',
    field_names=['pred_shape', 'pred_theta_mats', 'pred_phi', 'pred_delta_shape', 'pred_leaf',
                 'pred_uvd_jts', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
                 'pred_xyz_jts_17', 'pred_vertices', 'maxvals']
)
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


@SPPE.register_module
class HybrikResNetSMPL24(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HybrikResNetSMPL24, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = 24
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32

        backbone = ResNet

        self.preact = backbone(f"resnet{kwargs['NUM_LAYERS']}")
                
        # Imagenet pretrain model
        import torchvision.models as tm
        if kwargs['NUM_LAYERS'] == 101:
            ''' Load pretrained model '''
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(
            self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype,
            num_joints=self.num_joints
        )

        self.joint_pairs_24 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.leaf_pairs = ((0, 1), (3, 4))
        self.root_idx_24 = 0

        # mean shape
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(self.feature_channel, 1024)
#         self.drop1 = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.drop2 = nn.Dropout(p=0.5)
#         self.decshape = nn.Linear(1024, 10)
#         self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
#         self.decleaf = nn.Linear(1024, 5 * 4)  # rot_mat quat
        
        self.decjoint = nn.Linear(kwargs['RESNET']['LAYER_WIDTH'], 3)
        self.decshape = nn.Linear(kwargs['RESNET']['LAYER_WIDTH'], 10)
        self.decphi = nn.Linear(kwargs['RESNET']['LAYER_WIDTH'], 2)  # [cos(phi), sin(phi)]
        self.decleaf = nn.Linear(kwargs['RESNET']['LAYER_WIDTH'], 4)  # rot_mat quat
        
        self.resnet = [ResidualBlock(num_layers=kwargs['RESNET']['NUM_LAYERS'], 
                                     layer_width=kwargs['RESNET']['LAYER_WIDTH'],
                                     dropout=kwargs['RESNET']['DROPOUT'], 
                                     size_in=8*8*self.feature_channel)]
        
        self.resnet = self.resnet + [ResidualBlock(num_layers=kwargs['RESNET']['NUM_LAYERS'], 
                                                   layer_width=kwargs['RESNET']['LAYER_WIDTH'],
                                                   dropout=kwargs['RESNET']['DROPOUT'], 
                                                   size_in=kwargs['RESNET']['LAYER_WIDTH']) 
                                     for i in range(kwargs['RESNET']['NUM_BLOCKS']-1)]
        
        self.resnet = nn.ModuleList(self.resnet)
        self.projection = nn.Linear(kwargs['RESNET']['LAYER_WIDTH'], (24+23+5+1)*kwargs['RESNET']['LAYER_WIDTH'])
        
    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def uvd_to_cam(self, uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

        # remap uv coordinate to input space
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * self.width_dim * 4
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * self.height_dim * 4
        # remap d to mm
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)

        dz = uvd_jts_new[:, :, 2]

        # transform in-bbox coordinate to image coordinate
        uv_homo_jts = torch.cat(
            (uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]),
            dim=2)
        # batch-wise matrix multipy : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
        uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        # transform (u,v,1) to (x,y,z)
        cam_2d_homo = torch.cat(
            (uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]),
            dim=2)
        # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        # recover absolute z : (B,K) + (B,1)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        # multipy absolute z : (B,K,3) * (B,K,1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

        if return_relative:
            # (B,K,3) - (B,1,3)
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)

        xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)

        return xyz_jts

    def flip_uvd_coord(self, pred_jts, shift=False, flatten=True):
        num_joints = 24

        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        # flip
        if shift:
            pred_jts[:, :, 0] = - pred_jts[:, :, 0]
        else:
            pred_jts[:, :, 0] = -1 / self.width_dim - pred_jts[:, :, 0]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, num_joints * 3)

        return pred_jts

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def flip_leaf(self, pred_leaf):

        pred_leaf[:, :, 2] = -1 * pred_leaf[:, :, 2]
        pred_leaf[:, :, 3] = -1 * pred_leaf[:, :, 3]

        for pair in self.leaf_pairs:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_leaf[:, idx] = pred_leaf[:, inv_idx]

        return pred_leaf

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False):
        
        batch_size = x.shape[0]
        
        x0 = self.preact(x)
        
#         x0_flat = x0_pos.reshape(batch_size, self.feature_channel, -1).transpose(1, 2)
    
        h = x0.reshape(batch_size, -1)
        for rb in self.resnet:
            h = rb(h)
        h = self.projection(h)
        h = h.reshape(batch_size, 24+23+5+1, -1)
        
        pred_uvd_jts_24 = self.decjoint(h[:, :24])
        pred_phi = self.decphi(h[:, 24:24+23])
        pred_leaf = self.decleaf(h[:, 24+23:24+23+5])
        delta_shape = self.decshape(h[:, -1])
        pred_shape = delta_shape + self.init_shape.unsqueeze(0)
        
        maxvals = torch.ones((*pred_uvd_jts_24.shape[:2], 1), dtype=torch.float, device=x0.device)
        
        ############################################
        

#         x0 = self.avg_pool(x0)
#         x0 = x0.view(x0.size(0), -1)
#         init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)

#         xc = x0

#         xc = self.fc1(xc)
#         xc = self.drop1(xc)
#         xc = self.fc2(xc)
#         xc = self.drop2(xc)

#         delta_shape = self.decshape(xc)
#         pred_shape = delta_shape + init_shape
#         pred_phi = self.decphi(xc)
#         pred_leaf = self.decleaf(xc)

        if flip_item is not None:
            assert flip_output
            pred_uvd_jts_24_orig, pred_phi_orig, pred_leaf_orig, pred_shape_orig = flip_item

        if flip_output:
            pred_uvd_jts_24 = self.flip_uvd_coord(pred_uvd_jts_24, flatten=False, shift=True)
        if flip_output and flip_item is not None:
            pred_uvd_jts_24 = (pred_uvd_jts_24 + pred_uvd_jts_24_orig.reshape(batch_size, 24, 3)) / 2

        pred_uvd_jts_24_flat = pred_uvd_jts_24.reshape((batch_size, self.num_joints * 3))

        #  -0.5 ~ 0.5
        # Rotate back
        pred_xyz_jts_24 = self.uvd_to_cam(pred_uvd_jts_24[:, :24, :], trans_inv, intrinsic_param, joint_root, depth_factor)
        assert torch.sum(torch.isnan(pred_xyz_jts_24)) == 0, ('pred_xyz_jts_24', pred_xyz_jts_24)

        pred_xyz_jts_24 = pred_xyz_jts_24 - pred_xyz_jts_24[:, self.root_idx_24, :].unsqueeze(1)

        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        pred_leaf = pred_leaf.reshape(batch_size, 5, 4)

        if flip_output:
            pred_phi = self.flip_phi(pred_phi)
            pred_leaf = self.flip_leaf(pred_leaf)
        if flip_output and flip_item is not None:
            pred_phi = (pred_phi + pred_phi_orig) / 2
            pred_leaf = (pred_leaf + pred_leaf_orig) / 2
            pred_shape = (pred_shape + pred_shape_orig) / 2
            
        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_24.type(self.smpl_dtype) * 2,
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            leaf_thetas=pred_leaf.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_24_struct = output.joints.float() / 2
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = output.joints_from_verts.float() / 2
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        output = ModelOutput(
            pred_phi=pred_phi,
            pred_leaf=pred_leaf,
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_24_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17,
            pred_vertices=pred_vertices,
            maxvals=maxvals
        )
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):

        output = self.smpl(
            pose_axis_angle=gt_theta,
            betas=gt_beta,
            global_orient=None,
            return_verts=True
        )

        return output
