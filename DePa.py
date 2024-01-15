"""
Decoupling parameters
"""
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from skimage.io import imread
from tqdm import tqdm
import torch
import numpy as np
import mediapipe as mp
import os
import cv2
import trimesh
import matplotlib.pyplot as plt

def decouple(image_path: str, results_path: str, device='cuda'):

    # Get the image and converting it to a rgb so mediapipe can process it
    fn, ext = os.path.splitext(os.path.basename(image_path))
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8')

    #Using mediapipe to get the middle of the face
    #Initialization
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    #Getting Landmarks
    results = face_mesh.process(image_rgb)
    landmarks = results.multi_face_landmarks[0].landmark
    #getting Landmarks of the eyes
    left_eye_inner = np.array([landmarks[159].x, landmarks[159].y, landmarks[159].z])
    right_eye_inner = np.array([landmarks[386].x, landmarks[386].y, landmarks[386].z])
    #Calculation of the middle of the face
    eye_midpoint = (left_eye_inner + right_eye_inner) / 2

    eye_midpoint_pixel = (int(eye_midpoint[0] * image.shape[1]), int(eye_midpoint[1] * image.shape[0]))

    #Decoupling images
    right_img = image[:, :eye_midpoint_pixel[0], :]
    left_img = image[:, eye_midpoint_pixel[0]:, :]

    #Flipping images
    left_img_flip = np.flip(left_img, axis=1)
    right_img_flip = np.flip(right_img, axis=1)

    #Building new full faces for both sides
    left_img_mir = np.concatenate([left_img_flip, left_img], axis=1)
    right_img_mir= np.concatenate([right_img, right_img_flip], axis=1)

    left_mir_path = results_path + fn + "_left_mir.png"
    right_mir_path = results_path + fn + "_right_mir.png"

    cv2.imwrite(left_mir_path, left_img_mir)
    cv2.imwrite(right_mir_path, right_img_mir)

    # Using DECA to Build the FLAME Model for both sides and extract the parameters
    # Initialization
    deca = DECA(config = deca_cfg, device=device)

    #Get the data in the right format
    test_l = datasets.TestData(left_mir_path)
    test_r = datasets.TestData(right_mir_path)
    render_orig = True
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = False
    #Get the codedictionarys
    for i in tqdm(range(len(test_l))):
        name = test_l[i]['imagename']
        images_l = test_l[i]['image'].to(device)[None,...]
        with torch.no_grad(): 
            codedict_l = deca.encode(images_l)
            opdict_l, visdict_l = deca.decode(codedict_l) #tensor
            if render_orig:
                tform_l = test_l[i]['tform'][None, ...]
                tform_l = torch.inverse(tform_l).transpose(1,2).to(device)
                original_image_l = test_l[i]['original_image'][None, ...].to(device)
                _, orig_visdict_l = deca.decode(codedict_l, render_orig=True, original_image=original_image_l, tform=tform_l)    
                orig_visdict_l['inputs'] = original_image_l    
            
    deca.save_obj(os.path.join(results_path, fn + '_left.obj'), opdict_l)
    cv2.imwrite(os.path.join(results_path, fn + '_left_vis.png'), deca.visualize(visdict_l))

    #Get the codedictionarys
    for i in tqdm(range(len(test_r))):
        name = test_r[i]['imagename']
        images_r = test_r[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict_r = deca.encode(images_r)
            opdict_r, visdict_r = deca.decode(codedict_r) #tensor
            if render_orig:
                tform_r = test_r[i]['tform'][None, ...]
                tform_r = torch.inverse(tform_r).transpose(1,2).to(device)
                original_image_r = test_r[i]['original_image'][None, ...].to(device)
                _, orig_visdict_r = deca.decode(codedict_r, render_orig=True, original_image=original_image_r, tform=tform_r)    
                orig_visdict_r['inputs'] = original_image_r    
                
    deca.save_obj(os.path.join(results_path, fn + '_right.obj'), opdict_r)
    cv2.imwrite(os.path.join(results_path, fn + '_right_vis.png'), deca.visualize(visdict_r))
    
    #Build codedictionary for both sides
    codedict_decoupled = {
        'left': {
            'exp': codedict_l['exp'],
            'shape': codedict_l['shape'],
            'pose': codedict_l['pose']
        },
        'right': {
            'exp': codedict_r['exp'],
            'shape': codedict_r['shape'],
            'pose': codedict_r['pose']
        }
    }

    # Save codedictionary
    torch.save(codedict_decoupled, os.path.join(results_path, fn + '_codedict.pth'))
    
    vert_l = np.loadtxt(results_path + fn + '_left_vertice.txt')
    vert_r = np.loadtxt(results_path + fn + '_right_vertice.txt')
    faces = np.loadtxt(results_path + fn + '_right_faces.txt')

    vert_blended = binary_blending(vert_l, vert_r)
    
    vertex_colors = np.ones([vert_blended.shape[0], 4]) * [1.0, 1.0, 1.0, 1.0]
    tri_mesh = trimesh.Trimesh(vert_blended, faces, vertex_colors=vertex_colors)
    tri_mesh.export(results_path + fn + '_blended.obj')
    torch.cuda.empty_cache()
    print("Decoupling done")

def binary_blending(vertices_right, vertices_left):

    vertice_exp_lr = np.zeros((5023, 3))

    template_vertices = vertices_right
    vertice_max = template_vertices[:,0].max()
    vertice_min = template_vertices[:,0].min()
    vertice_diff = vertice_max - vertice_min

    for i in range(0, 5023):
        rel_dist = (template_vertices[i, 0] - vertice_min) / vertice_diff
        vertice_exp_lr[i] = vertices_right[i]+0.1*vertices_left[i] if rel_dist > 0.5 else vertices_left[i]

    return vertice_exp_lr





