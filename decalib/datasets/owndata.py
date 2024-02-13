import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from tqdm import tqdm
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class OwnDataset(Dataset):
    def __init__(self, K, image_size, scale, trans_scale = 0, isTemporal=False, isEval=False, isSingle=False):
        '''
        K must be less than 6
        '''
        self.K = 6
        self.image_size = image_size
        basefolder = '/home/dietrich/Testing/DECA/Dataset/Probanden'
        self.imagefolder = os.path.join(basefolder, 'Probanden')
        self.kptfolder = os.path.join(basefolder, 'lmks_train')
        self.shapefolder = os.path.join(basefolder, 'shape_train') # _mean
        # self.segfolder = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_seg/test_crop_size_400_batch'
        # hq:
        # Does not exist; doing it without data cleaning
        datafile = os.path.join(basefolder, 'data_names_9_per_individual_train.npy')
        if isEval:
            datafile = os.path.join(basefolder, 'data_names_9_per_individual_valid.npy')
        self.data_lines = np.load(datafile).astype('str')

        self.isTemporal = isTemporal
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        images_list = []; kpt_list = []; mask_list = []; shape_list = []

        for i in range(self.K):
            name = self.data_lines[idx, i]
            image_path = os.path.join(self.imagefolder, name + '.jpg')  
            #seg_path = os.path.join(self.segfolder, name + '.npy')  
            kpt_path = os.path.join(self.kptfolder, name + '_68kpts.npy')
            if self.shapefolder.endswith('mean'):
                shape_path = os.path.join(self.shapefolder, name.split('/')[0]  + '/shape_mean.npy')
            else:
                shape_path = os.path.join(self.shapefolder, name + '.npy')
                                            
            image = imread(image_path)/255.0
            kpt = np.load(kpt_path)[:,:2]
            shape = np.load(shape_path)
            #mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop 
            cropped_imgs = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            #cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpts = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpts[:,:2] = cropped_kpts[:,:2]/self.image_size * 2  - 1

            images_list.append(cropped_imgs.transpose(2,0,1))
            kpt_list.append(cropped_kpts)
            #mask_list.append(cropped_mask)
            shape_list.append(shape)

        ###
        imgs_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        kpts_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32) #K,224,224,3
        shapearray = torch.from_numpy(np.array(shape_list)).type(dtype = torch.float32) #K,100

        if self.isSingle:
            imgs_array = imgs_array.squeeze()
            kpts_array = kpts_array.squeeze()
            mask_array = mask_array.squeeze()
            shapearray = shapearray.squeeze()
                    
        data_dict = {
            'image':    imgs_array,
            'landmark': kpts_array,
            'mask':     mask_array,
            'shape':    shapearray
        }
        
        return data_dict
    
    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform
    
    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask