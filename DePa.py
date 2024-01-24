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
from decalib.models.encoders import ResnetEncoder
import numpy as np
from decalib.datasets import datasets 
import torch
import os
from decalib.utils import util

def decompose_code(concatCode, num_dict__):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = [('shape',) 'tex', 'exp', 'pose', 'cam', 'light']
        Copied from deca
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

def encode(image_path: str, model_path='/home/dietrich/Testing/DECA/DECA/data/deca_model.tar'):
    """Encode an image to obtain parameters.

    Args:
        image_path (str): Path to the input image.
        model_path (str, optional): Path to the DECA model checkpoint (default is '/home/dietrich/Testing/DECA/DECA/data/deca_model.tar').

    Returns:
        dict: A dictionary containing parameters and the input image.
    """
    
    n_shape = 100
    n_tex = 50
    n_exp = 50
    n_pose = 6
    n_cam = 3
    n_light = 27

    num_params = n_shape+n_tex+n_exp+n_pose+n_cam+n_light
    E_flame  = ResnetEncoder(outsize=num_params).to('cuda')

    param_dict = {'shape': n_shape, 'tex': n_tex, 'exp': n_exp, 'pose': n_pose, 'cam': n_cam, 'light': n_light}

    checkpoint = torch.load(model_path)
    util.copy_state_dict(E_flame.state_dict(), checkpoint['E_flame'])
    E_flame.eval()

    testdata__ = datasets.TestData(image_path)

    for i in range(len(image_path)):
        imgs_batch = testdata__[0]['image'].to('cuda')[None,...]
        with torch.no_grad():
                parameters = E_flame(imgs_batch)
        code_dict__ = decompose_code(parameters, param_dict)
        code_dict__['images'] = imgs_batch
            
    return code_dict__

def decouple(image_path: str, results_path: str, device='cuda'):
    """ Taking an image and decouple it in left an right side + generating flipped images.

    Args:
        image_path (str): Path to the input image.
        results_path (str): Path to save the decoupled images.
        device (str, optional): Device for processing (default is 'cuda').

    Returns:
        tuple: A tuple containing:
            - fn (str): The base filename of the input image.
            - left_mir_path (str): Path to the decoupled and mirrored left side image.
            - right_mir_path (str): Path to the decoupled and mirrored right side image.

    """
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
         
    print("Decoupling done")

    return fn, left_mir_path, right_mir_path

def binary_blending(vertices_right, vertices_left):
    """Blend two sets of 3D vertices using a binary mask.

    Args:
        vertices_right (np.ndarray): 3D vertices for the right side.
        vertices_left (np.ndarray): 3D vertices for the left side.

    Returns:
        np.ndarray: Blended 3D vertices.
    """

    vertice_exp_lr = np.zeros((5023, 3))

    template_vertices = vertices_right
    vertice_max = template_vertices[:,0].max()
    vertice_min = template_vertices[:,0].min()
    vertice_diff = vertice_max - vertice_min

    for i in range(0, 5023):
        rel_dist = (template_vertices[i, 0] - vertice_min) / vertice_diff
        vertice_exp_lr[i] = vertices_right[i]+0.1*vertices_left[i] if rel_dist > 0.5 else vertices_left[i]

    return vertice_exp_lr

def decode_wo_shape(fn:str, image_path:str, results_path: str, pretrained_modelpath=None):
    # Using DECA to Build the FLAME Model for both sides and extract the parameters
    # Initialization
    if pretrained_modelpath is not None:
        deca_cfg.pretrained_modelpath = pretrained_modelpath

    deca_cfg.train_mode = 'without_shape'
    deca = DECA()
    #Get the data in the right format
    test = datasets.TestData(image_path)
    render_orig = True
    #Get the codedictionarys
    for i in tqdm(range(len(test))):
        name = test[i]['imagename']
        images = test[i]['image'].to('cuda')[None,...]
        with torch.no_grad(): 
            codedict = deca.encode(images)
            shape_params = get_shape_params(image_path)
            opdict, visdict = deca.decode(codedict, shape_params=shape_params) #tensor
            if render_orig:
                tform = test[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to('cuda')
                original_image = test[i]['original_image'][None, ...].to('cuda')
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform, shape_params=shape_params)    
                orig_visdict['inputs'] = original_image    
            
    deca.save_obj(os.path.join(results_path, fn + '.obj'), opdict)
    cv2.imwrite(os.path.join(results_path, fn + '_vis.png'), deca.visualize(visdict))

    # Save codedictionary
    torch.save(codedict, os.path.join(results_path, fn + '_codedict.pth'))

    torch.cuda.empty_cache()

    return codedict

def encode_wo_shape(image_path:str, pretrained_modelpath=None):
    n_tex = 50
    n_exp = 50
    n_pose = 6
    n_cam = 3
    n_light = 27

    num_params = n_tex+n_exp+n_pose+n_cam+n_light
    E_flame  = ResnetEncoder(outsize=num_params).to('cuda')

    param_dict = {'tex': n_tex, 'exp': n_exp, 'pose': n_pose, 'cam': n_cam, 'light': n_light}

    checkpoint = torch.load(pretrained_modelpath)
    util.copy_state_dict(E_flame.state_dict(), checkpoint['E_flame'])
    E_flame.eval()

    testdata__ = datasets.TestData(image_path)

    for i in range(len(image_path)):
        imgs_batch = testdata__[0]['image'].to('cuda')[None,...]
        with torch.no_grad():
                parameters = E_flame(imgs_batch)
        code_dict__ = decompose_code(parameters, param_dict)
        code_dict__['images'] = imgs_batch
    
    return code_dict__

def get_shape_params(image_path: str, device='cuda'):
    """Function for getting the shape paramters of an image
    """
    deca_cfg.train_mode = ''
    deca = DECA()
    testdata__ = datasets.TestData(image_path)
    for i in range(len(image_path)):
        imgs_batch = testdata__[0]['image'].to('cuda')[None,...]
        with torch.no_grad(): 
            codedict = deca.encode(imgs_batch)
            
    return codedict['shape']

def get_results(base_folder:str, output_base_folder:str, get_visual = False, pretrained_modelpath = None):
    # Erstelle den Ausgabeordner, wenn er nicht existiert
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # Gehe durch alle Ordner im Ausgangsverzeichnis
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        # Überprüfe, ob es sich um einen Ordner handelt
        if os.path.isdir(folder_path):
            # Erstelle den Ausgabeordner für diesen Ordner, wenn er nicht existiert
            output_folder_path = os.path.join(output_base_folder, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # Durchlaufe alle Dateien im Ordner
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # Überprüfe, ob es sich um eine Bilddatei handelt
                if file_name.lower().endswith(('.jpg', '.JPG')):
                    fn, ext = os.path.splitext(os.path.basename(file_name))
                    # Erstelle den Ergebnis-Pfad für diese Datei
                    result_path = os.path.join(output_folder_path, fn + "/")
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                    
                    # Führe die DePa.decouple-Funktion auf die Bilddatei aus
                    fn, left_mir, right_mir = decouple(file_path, result_path)
                    if get_visual:
                        decode_wo_shape(fn + "_left_", left_mir, result_path)
                        decode_wo_shape(fn + "_right_", right_mir, result_path)

                    left_shape = encode(left_mir)
                    right_shape = encode(right_mir)
                    
                    left_codedict = encode_wo_shape(left_mir, pretrained_modelpath)
                    right_codedict = encode_wo_shape(left_mir, pretrained_modelpath)

                    

                    codedict_decoupled = {
                        'left': {
                            'shape': left_shape['shape'],
                            'exp': left_codedict['exp'],
                            'pose': left_codedict['pose']
                        },
                        'right': {
                            'shape': right_shape['shape'],
                            'exp': right_codedict['exp'],
                            'pose': right_codedict['pose']
                        }

                    }

                    torch.save(codedict_decoupled, os.path.join(result_path,fn+"_codedict.pth"))