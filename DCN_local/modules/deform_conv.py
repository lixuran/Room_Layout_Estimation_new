#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair
import numpy as np
from functions.deform_conv_func import DeformConvFunction

class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        return DeformConvFunction.apply(input, offset,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_DeformConv = DeformConvFunction.apply

class DeformConvPack(DeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.offset= None
        self.offset_set =0

    def forward(self, input):
        
        pano_W=input.shape[3]
        pano_H=input.shape[2]
        bs=input.shape[0]
        #stop gradient here
        #print(self.kernel_size[0],self.kernel_size[1])
        #print(type(self.kernel_size[1]))
        if (self.offset_set == 0) :
            self.offset = distortion_aware_map(pano_W, pano_H, self.kernel_size[0], self.kernel_size[1], s_width=self.stride[0], s_height=self.stride         [1],bs=bs)
            self.offset_set =1
        
        return DeformConvFunction.apply(input, self.offset, 
                                          self.weight, 
                                          self.bias, 
                                          self.stride, 
                                          self.padding, 
                                          self.dilation, 
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)
    
# copied from CFL
#@staticmethod
def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
 
#@staticmethod
def equi_coord(pano_W,pano_H,k_W,k_H,u,v): 
        """ contribution by cfernandez and jmfacil """
        fov_w = k_W * np.deg2rad(360./float(pano_W))
        focal = (float(k_W)/2) / np.tan(fov_w/2)
        c_x = 0
        c_y = 0

        u_r, v_r = u, v 
        u_r, v_r = u_r-float(pano_W)/2.,v_r-float(pano_H)/2.
        phi, theta = u_r/(pano_W) * (np.pi) *2, -v_r/(pano_H) * (np.pi)

        ROT = rotation_matrix((0,1,0),phi)
        ROT = np.matmul(ROT,rotation_matrix((1,0,0),theta))#np.eye(3)

        h_range = np.array(range(k_H))
        w_range = np.array(range(k_W))
        w_ones = (np.ones(k_W))
        h_ones = (np.ones(k_H))
        h_grid = np.matmul(np.expand_dims(h_range,-1),np.expand_dims(w_ones,0))+0.5-float(k_H)/2
        w_grid = np.matmul(np.expand_dims(h_ones,-1),np.expand_dims(w_range,0))+0.5-float(k_W)/2
        
        K=np.array([[focal,0,c_x],[0,focal,c_y],[0.,0.,1.]])
        inv_K = np.linalg.inv(K)
        rays = np.stack([w_grid,h_grid,np.ones(h_grid.shape)],0)
        rays = np.matmul(inv_K,rays.reshape(3,k_H*k_W))
        rays /= np.linalg.norm(rays,axis=0,keepdims=True)
        rays = np.matmul(ROT,rays)
        rays=rays.reshape(3,k_H,k_W)
        
        phi = np.arctan2(rays[0,...],rays[2,...])
        theta = np.arcsin(np.clip(rays[1,...],-1,1))
        x = (pano_W)/(2.*np.pi)*phi +float(pano_W)/2.
        y = (pano_H)/(np.pi)*theta +float(pano_H)/2.
        
        roi_y = h_grid+v_r +float(pano_H)/2.
        roi_x = w_grid+u_r +float(pano_W)/2.

        new_roi_y = (y) 
        new_roi_x = (x) 

        offsets_x = (new_roi_x - roi_x)
        offsets_y = (new_roi_y - roi_y)

        return offsets_x, offsets_y

#@staticmethod
def equi_coord_fixed_resoltuion(pano_W,pano_H,k_W,k_H,u,v,pano_Hf = -1, pano_Wf=-1): 
        """ contribution by cfernandez and jmfacil """
        pano_Hf = pano_H if pano_Hf<=0 else pano_H/pano_Hf
        pano_Wf = pano_W if pano_Wf<=0 else pano_W/pano_Wf
        fov_w = k_W * np.deg2rad(360./float(pano_Wf))
        focal = (float(k_W)/2) / np.tan(fov_w/2)
        c_x = 0
        c_y = 0

        u_r, v_r = u, v 
        u_r, v_r = u_r-float(pano_W)/2.,v_r-float(pano_H)/2.
        phi, theta = u_r/(pano_W) * (np.pi) *2, -v_r/(pano_H) * (np.pi)

        ROT = rotation_matrix((0,1,0),phi)
        ROT = np.matmul(ROT,rotation_matrix((1,0,0),theta))#np.eye(3)

        h_range = np.array(range(k_H))
        w_range = np.array(range(k_W))
        w_ones = (np.ones(k_W))
        h_ones = (np.ones(k_H))
        h_grid = np.matmul(np.expand_dims(h_range,-1),np.expand_dims(w_ones,0))+0.5-float(k_H)/2
        w_grid = np.matmul(np.expand_dims(h_ones,-1),np.expand_dims(w_range,0))+0.5-float(k_W)/2
        
        K=np.array([[focal,0,c_x],[0,focal,c_y],[0.,0.,1.]])
        inv_K = np.linalg.inv(K)
        rays = np.stack([w_grid,h_grid,np.ones(h_grid.shape)],0)
        rays = np.matmul(inv_K,rays.reshape(3,k_H*k_W))
        rays /= np.linalg.norm(rays,axis=0,keepdims=True)
        rays = np.matmul(ROT,rays)
        rays=rays.reshape(3,k_H,k_W)
        
        phi = np.arctan2(rays[0,...],rays[2,...])
        theta = np.arcsin(np.clip(rays[1,...],-1,1))
        x = (pano_W)/(2.*np.pi)*phi +float(pano_W)/2.
        y = (pano_H)/(np.pi)*theta +float(pano_H)/2.
        
        roi_y = h_grid+v_r +float(pano_H)/2.
        roi_x = w_grid+u_r +float(pano_W)/2.

        new_roi_y = (y) 
        new_roi_x = (x) 

        offsets_x = (new_roi_x - roi_x)
        offsets_y = (new_roi_y - roi_y)

        return offsets_x, offsets_y

#@staticmethod
def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width = 1, s_height = 1,bs = 16):
        """ contribution by cfernandez and jmfacil """
        # pano_W width of the input tensor
        # pano_H hight of the input tensor 
        # k_w kernel width == k_H kernel height == 3?
        # s_width stride width == 1
        # s_height stride height == 1
        # bs batch size probably should be smaller?
        n=1
        offset = np.zeros(shape=[pano_H,pano_W,k_H*k_W*2])
        #print(offset.shape)
        
        for v in range(0, pano_H, s_height): 
            for u in range(0, pano_W, s_width): 
                offsets_x, offsets_y = equi_coord_fixed_resoltuion(pano_W,pano_H,k_W,k_H,u,v,1,1)
                offsets = np.concatenate((np.expand_dims(offsets_y,-1),np.expand_dims(offsets_x,-1)),axis=-1)
                total_offsets = offsets.flatten().astype("float32")
                offset[v,u,:] = total_offsets
        #print("second offset" + offset.shape)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        offset = torch.tensor(offset,device=device)
        offset = offset.unsqueeze(0)
        offset = torch.cat([offset for _ in range(bs)],0)
        offset = offset.type(torch.float)
        # to nchw
        offset = torch.transpose(offset,1,2)    
        offset = torch.transpose(offset,1,3)
        offset.requires_grad=False  
        return offset
        
###        
