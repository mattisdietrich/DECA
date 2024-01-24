# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from .utils.renderer import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.config import get_config
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.tensor_cropper import transform_points
from .datasets import datasets
from .utils.config import cfg as config_default
torch.backends.cudnn.benchmark = True
from IPython.display import display
import multiprocessing as mp

class DECA(nn.Module):
    def __init__(self, config_build=None, device='cuda'):
        super(DECA, self).__init__()
        if config_build is None:
            self.config = config_default
        else:
            self.config = config_build
        if self.config.train_mode == 'without_shape':
            self.train_bool = True
        else:
            self.train_bool = False


        self.device____ = device
        self.image_size = self.config.dataset.image_size
        self.uv_size___ = self.config.model.uv_size

        self._create_model__(self.config.model)
        self._setup_renderer(self.config.model)

    def _setup_renderer(self, model_cfg_):
        set_rasterizer(self.config.rasterizer_type)
        self.render = SRenderY(self.image_size, obj_filename=model_cfg_.topology_path, uv_size=model_cfg_.uv_size, rasterizer_type=self.config.rasterizer_type).to(self.device____)
        # face mask for rendering details
        msk_render = imread(model_cfg_.face_eye_mask_path).astype(np.float32)/255.
        msk_render = torch.from_numpy(msk_render[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(msk_render, [model_cfg_.uv_size, model_cfg_.uv_size]).to(self.device____)
        msk_render = imread(model_cfg_.face_mask_path).astype(np.float32)/255.
        msk_render = torch.from_numpy(msk_render[:,:,0])[None,None,:,:].contiguous()
        self.uv_face___ = F.interpolate(msk_render, [model_cfg_.uv_size, model_cfg_.uv_size]).to(self.device____)
        # displacement correction
        fixed_dis_ = np.load(model_cfg_.fixed_displacement_path)
        self.fix_uv_dis = torch.tensor(fixed_dis_).float().to(self.device____)
        # mean texture
        mean_text_ = imread(model_cfg_.mean_tex_path).astype(np.float32)/255.0
        mean_text_ = torch.from_numpy(mean_text_.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_text_ = F.interpolate(mean_text_, [model_cfg_.uv_size, model_cfg_.uv_size]).to(self.device____)
        # dense mesh template, for save detail mesh
        self.dense_temp = np.load(model_cfg_.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def _create_model__(self, model_cfg_):
        # set up parameters
        if not self.train_bool:
            self.num_params = model_cfg_.n_shape+model_cfg_.n_tex+model_cfg_.n_exp+model_cfg_.n_pose+model_cfg_.n_cam+model_cfg_.n_light
            self.num_list__ = [model_cfg_.n_shape, model_cfg_.n_tex, model_cfg_.n_exp, model_cfg_.n_pose, model_cfg_.n_cam, model_cfg_.n_light]
            self.param_dict = {i:model_cfg_.get('n_' + i) for i in model_cfg_.param_list}
        else:
            self.num_params = model_cfg_.n_tex+model_cfg_.n_exp+model_cfg_.n_pose+model_cfg_.n_cam+model_cfg_.n_light
            self.num_list__ = [model_cfg_.n_tex, model_cfg_.n_exp, model_cfg_.n_pose, model_cfg_.n_cam, model_cfg_.n_light]
            self.param_dict = {i:model_cfg_.get('n_' + i) for i in model_cfg_.param_list_wo_shape}
        
        
        self.num_detail = model_cfg_.n_detail
        self.num_cond__ = model_cfg_.n_exp + 3 # exp + jaw pose
        
        # encoders
        self.E_flame  = ResnetEncoder(outsize=self.num_params).to(self.device____) 
        self.E_detail = ResnetEncoder(outsize=self.num_detail).to(self.device____)

        # decoders
        self.flame = FLAME(model_cfg_).to(self.device____)
        if model_cfg_.use_tex:
            self.flametex = FLAMETex(model_cfg_).to(self.device____)
        self.D_detail = Generator(latent_dim=self.num_detail+self.num_cond__, out_channels=1, out_scale=model_cfg_.max_z, sample_mode = 'bilinear').to(self.device____)
        # resume model
        model_path = self.config.pretrained_modelpath
        if os.path.exists(model_path):
            #print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        else:
            print(f'please check model path: {model_path}')
            # exit()
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, concatCode, num_dict__):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = [('shape',) 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict_ = {}
        start_ind_ = 0
        for key in num_dict__:
            end_ind___      = start_ind_+int(num_dict__[key])
            code_dict_[key] = concatCode[:, start_ind_:end_ind___]
            start_ind_      = end_ind___
            if key == 'light':
                code_dict_[key] = code_dict_[key].reshape(code_dict_[key].shape[0], 9, 3)
        return code_dict_

    def displacement2normal(self, uv_z_map__, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z_map__.shape[0]
        coarsevert = self.render.world2uv(coarse_verts).detach()
        coarsenorm = self.render.world2uv(coarse_normals).detach()
    
        uv_z_map__ = uv_z_map__*self.uv_face_eye_mask
        detailvert = coarsevert + uv_z_map__*coarsenorm + self.fix_uv_dis[None,None,:,:]*coarsenorm.detach()
        dense_vert = detailvert.permute(0,2,3,1).reshape([batch_size, -1, 3])
        detailnorm = util.vertex_normals(dense_vert, self.render.dense_faces.expand(batch_size, -1, -1))
        detailnorm = detailnorm.reshape([batch_size, coarsevert.shape[2], coarsevert.shape[3], 3]).permute(0,3,1,2)
        detailnorm = detailnorm*self.uv_face_eye_mask + coarsenorm*(1.-self.uv_face_eye_mask)
        return detailnorm

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:,:,2:] < 0.1).float()
        return vis68

    # @torch.no_grad()
    def encode(self, imgs_batch, use_detail=True):
        if use_detail:
            # use_detail is for training detail model, need to set coarse model as eval mode
            with torch.no_grad():
                parameters = self.E_flame(imgs_batch)
        else:
            parameters = self.E_flame(imgs_batch)

        code_dict__ = self.decompose_code(parameters, self.param_dict)
        code_dict__['images'] = imgs_batch
        if use_detail:
            code_detail_ = self.E_detail(imgs_batch)
            code_dict__['detail'] = code_detail_
        if self.config.model.jaw_type == 'euler':
            code_pose__ = code_dict__['pose']
            euler_jaw__ = code_pose__[:,3:].clone() # x for yaw (open mouth), y for pitch (left ang right), z for roll
            code_pose__[:,3:] = batch_euler2axis(euler_jaw__)
            code_dict__['pose'] = code_pose__
            code_dict__['euler_jaw_pose'] = euler_jaw__
        return code_dict__

    # @torch.no_grad()
    def decode(self, code_dict__, rendering=True, iddict=None, vis_lmk=True, return_vis=True, use_detail=True,
                render_orig=False, original_image=None, tform=None, shape_params=None):
        
        imgs_batch = code_dict__['images']
        batch_size = imgs_batch.shape[0]
        
        ## decode
        if not self.train_bool:
            verts, lmks_2dim_, lmks_3dim_ = self.flame(shape_params=code_dict__['shape'], expression_params=code_dict__['exp'], pose_params=code_dict__['pose'])
        else:
            verts, lmks_2dim_, lmks_3dim_ = self.flame(shape_params=shape_params, expression_params=code_dict__['exp'], pose_params=code_dict__['pose'])

        
        if self.config.model.use_tex:
            albedo = self.flametex(code_dict__['tex'])
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size___, self.uv_size___], device=imgs_batch.device) 
        landmarks3d_world = lmks_3dim_.clone()

        ## projection
        lmks_2dim_ = util.batch_orth_proj(lmks_2dim_, code_dict__['cam'])[:,:,:2]; lmks_2dim_[:,:,1:] = -lmks_2dim_[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        lmks_3dim_ = util.batch_orth_proj(lmks_3dim_, code_dict__['cam']); lmks_3dim_[:,:,1:] = -lmks_3dim_[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        tran_verts = util.batch_orth_proj(verts, code_dict__['cam']); tran_verts[:,:,1:] = -tran_verts[:,:,1:]
        optic_dict = {
            'verts': verts,
            'trans_verts': tran_verts,
            'landmarks2d': lmks_2dim_,
            'landmarks3d': lmks_3dim_,
            'landmarks3d_world': landmarks3d_world,
        }

        ## rendering
        if return_vis and render_orig and original_image is not None and tform is not None:
            pts_scale_ = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            # import ipdb; ipdb.set_trace()
            tran_verts = transform_points(tran_verts, tform, pts_scale_, [h, w])
            lmks_2dim_ = transform_points(lmks_2dim_, tform, pts_scale_, [h, w])
            lmks_3dim_ = transform_points(lmks_3dim_, tform, pts_scale_, [h, w])
            background = original_image
            imgs_batch = original_image
        else:
            h, w       = self.image_size, self.image_size
            background = None

        if rendering:
            # ops = self.render(verts, trans_verts, albedo, codedict['light'])
            ops_render = self.render(verts, tran_verts, albedo, h=h, w=w, background=background)
            ## output
            optic_dict['grid'] = ops_render['grid']
            optic_dict['rendered_images'] = ops_render['images']
            optic_dict['alpha_images'] = ops_render['alpha_images']
            optic_dict['normal_images'] = ops_render['normal_images']
        
        if self.config.model.use_tex:
            optic_dict['albedo'] = albedo
              
        if use_detail:
            uv_z_map__ = self.D_detail(torch.cat([code_dict__['pose'][:,3:], code_dict__['exp'], code_dict__['detail']], dim=1))
            if iddict is not None:
                uv_z_map__ = self.D_detail(torch.cat([iddict['pose'][:,3:], iddict['exp'], code_dict__['detail']], dim=1))
            detailnorm = self.displacement2normal(uv_z_map__, verts, ops_render['normals'])
            uv_shading = self.render.add_SHlight(detailnorm, code_dict__['light'])
            uv_texture = albedo*uv_shading

            optic_dict['uv_texture'] = uv_texture 
            optic_dict['normals'] = ops_render['normals']
            optic_dict['uv_detail_normals'] = detailnorm
            optic_dict['displacement_map'] = uv_z_map__+self.fix_uv_dis[None,None,:,:]
        
        if vis_lmk:
            lmks3d_vis = self.visofp(ops_render['transformed_normals'])#/self.image_size
            lmks_3dim_ = torch.cat([lmks_3dim_, lmks3d_vis], dim=2)
            optic_dict['landmarks3d'] = lmks_3dim_

        if return_vis:
            ## render shape
            shapeimgs, _, grid, alpha_imgs = self.render.render_shape(verts, tran_verts, h=h, w=w, images=background, return_grid=True)
            detnormimg = F.grid_sample(detailnorm, grid, align_corners=False)*alpha_imgs
            shapdetimg = self.render.render_shape(verts, tran_verts, detail_normal_images=detnormimg, h=h, w=w, images=background)
            
            ## extract texture
            ## TODO: current resolution 256x256, support higher resolution, and add visibility
            uv_p_verts = self.render.world2uv(tran_verts)
            uv_gtverts = F.grid_sample(imgs_batch, uv_p_verts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear', align_corners=False)
            if self.config.model.use_tex:
                ## TODO: poisson blending should give better-looking results
                if self.config.model.extract_tex:
                    uv_text_gt = uv_gtverts[:,:3,:,:]*self.uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-self.uv_face_eye_mask))
                else:
                    uv_text_gt = uv_texture[:,:3,:,:]
            else:
                uv_text_gt = uv_gtverts[:,:3,:,:]*self.uv_face_eye_mask + (torch.ones_like(uv_gtverts[:,:3,:,:])*(1-self.uv_face_eye_mask)*0.7)
            
            optic_dict['uv_texture_gt'] = uv_text_gt
            visualdict = {
                'inputs': imgs_batch, 
                'landmarks2d': util.tensor_vis_landmarks(imgs_batch, lmks_2dim_),
                'landmarks3d': util.tensor_vis_landmarks(imgs_batch, lmks_3dim_),
                'shape_images': shapeimgs,
                'shape_detail_images': shapdetimg
            }
            if self.config.model.use_tex:
                visualdict['rendered_images'] = ops_render['images']

            return optic_dict, visualdict

        else:
            return optic_dict

    def visualize(self, visdict, size=224, dim=2):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim==2
        grids = {}
        for key in visdict:
            _,_,h,w = visdict[key].shape
            if dim == 2:
                new_h = size; new_w = int(w*size/h)
            elif dim == 1:
                new_h = int(h*size/w); new_w = size
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu())
        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        return grid_image
    
    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices__ = opdict['verts'][i].cpu().numpy()
        faces_vert = self.render.faces[0].cpu().numpy()
        
        texture___ = util.tensor2image(opdict['uv_texture_gt'][i])
        uv_coords_ = self.render.raw_uvcoords[0].cpu().numpy()
        uv_faces__ = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
        util.write_obj(filename, vertices__, faces_vert, 
                        texture=texture___, 
                        uvcoords=uv_coords_, 
                        uvfaces=uv_faces__, 
                        normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture___ = texture___[:,:,[2,1,0]]
        normals___ = opdict['normals'][i].cpu().numpy()
        displacMap = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
        dense_vert, dense_colo, dense_face = util.upsample_mesh(vertices__, normals___, faces_vert, displacMap, texture___, self.dense_temp)
        util.write_obj(filename.replace('.obj', '_detail.obj'), 
                        dense_vert, 
                        dense_face,
                        colors = dense_colo,
                        inverse_face_order=True)
        np.savetxt(filename.replace('.obj', '_vertice.txt'), vertices__)
        np.savetxt(filename.replace('.obj', '_faces.txt'), faces_vert)
        
    
    def run(self, imagepath, iscrop=True):
        ''' An api for running deca given an image path
        '''
        testdata__ = datasets.TestData(imagepath)
        imgs_batch = testdata__[0]['image'].to(self.device____)[None,...]
        cide_dict_ = self.encode(imgs_batch)
        opdic_dict, visualdict = self.decode(cide_dict_)
        return cide_dict_, opdic_dict, visualdict

    def model_dict(self):
        """Returns the Dictionaries

        Returns:
            Dictionary: Dictionaries with the checkpoints for Flame and Detail 
        """
        return {
            'E_flame':  self.E_flame.state_dict(),
            'E_detail': self.E_detail.state_dict(),
            'D_detail': self.D_detail.state_dict()
        }
