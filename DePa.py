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
from decalib.utils import util
from sklearn.manifold import TSNE
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import re
from matplotlib.lines import Line2D


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

def get_results(base_folder:str, output_base_folder:str, get_visual = False, pretrained_modelpath = None, only_decouple=False):
    """
    Process images in the given base folder and save results, espacially codedict in the output base folder.

    Args:
        base_folder (str): Path to the input images base folder.
        output_base_folder (str): Path to the folder where results will be saved.
        get_visual (bool, optional): If True, generate visualizations using DECA (default is False).
        pretrained_modelpath (str, optional): Path to the pretrained DECA model checkpoint.

    Returns:
        None
    """
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
                        run_deca(left_mir, result_path, only_encode=False, wo_shape=True, pretrained_modelpath=pretrained_modelpath)
                        run_deca(right_mir, result_path, only_encode=False, wo_shape=True, pretrained_modelpath=pretrained_modelpath)

                    if not only_decouple:
                        left_shape = run_deca(left_mir, result_path, only_encode=True, wo_shape=False, pretrained_modelpath=None)
                        right_shape = run_deca(right_mir, result_path, only_encode=True, wo_shape=False, pretrained_modelpath=None)
                        
                        left_codedict = run_deca(left_mir, result_path, only_encode=True, wo_shape=True, pretrained_modelpath=pretrained_modelpath)
                        right_codedict = run_deca(right_mir, result_path, only_encode=True, wo_shape=True, pretrained_modelpath=pretrained_modelpath)

                        

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

                        torch.save(codedict_decoupled, os.path.join(result_path,fn+"_dec_codedict.pth"))

def visualization_exp_shape(expression_indices, expressionnames, parameter_list):
    """
    Visualize expressions and shapes using t-SNE with interactive sliders.

    Parameters:
    - expression_indices (list): List of indices representing expressions.
    - expressionnames (list): List of expression names corresponding to expression_indices.
    - parameter_list (list): List of parameter arrays for expressions and shapes.

    Returns:
    None
    """
    # Define color mapping for expressions
    expression_colors = {
    "Neutral": "cyan",
    "Stirn runzeln": "orange",
    "Augen zu": "lightgreen",
    "Augen doll zu": "green",
    "Nase rümpfen": "purple",
    "Lächeln": "mistyrose",
    "Lächeln doll": "red",
    "Mund spitzen": "gray",
    "Wangen aufblasen":"yellow",
    "Zähne fletschen":"brown",
    "Mundwinkel hoch":"blue",
    "Mundwinkel runter": "lightblue",
    }

    # Extract parameters for expressions and shapes
    ind = expression_indices
    exp_params_extract = np.concatenate([parameter_list[0][ind], parameter_list[1][ind]], axis=-1)
    exp_params_combined = exp_params_extract.reshape(-1, exp_params_extract.shape[-1])

    shape_params_extract = np.concatenate([parameter_list[4][ind], parameter_list[5][ind]], axis=-1)
    shape_params_combined = shape_params_extract.reshape(-1, shape_params_extract.shape[-1])

    # Map expressions to colors
    colors = np.zeros(exp_params_combined.shape[0], dtype=object)
    for i in range(exp_params_combined.shape[0]):
        for j in range(len(ind)):
            for h in range(len(parameter_list[0][0])):
                if np.array_equal(exp_params_combined[i], exp_params_extract[j, h]):
                    colors[i] = expression_colors[expressionnames[j]]
                    break

    def update_plot(perplexity_1, perplexity_2):
        # Compute t-SNE
        tsne_1 = TSNE(n_components=2, perplexity=perplexity_1)
        X_tsne_1 = tsne_1.fit_transform(exp_params_combined)

        tsne_2 = TSNE(n_components=2, perplexity=perplexity_2)
        X_tsne_2 = tsne_2.fit_transform(shape_params_combined)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        scatter_1 = axes[0].scatter(X_tsne_1[:, 0], X_tsne_1[:, 1], c=colors)
        axes[0].set_title(f'Expression TSNE with Perplexity={perplexity_1}')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        legend_1 = axes[0].legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=expression_colors[expressionnames[i]], markersize=10)
            for i in range(len(expressionnames))
                ], labels=expressionnames)
        
        scatter_2 = axes[1].scatter(X_tsne_2[:, 0], X_tsne_2[:, 1], c=colors)
        axes[1].set_title(f'Shape TSNE with Perplexity={perplexity_2}')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.savefig('tsne_plot.png')
        plt.show()
        
    # Create sliders for perplexity values
    perplexity_slider_1 = widgets.IntSlider(value=5, min=1, max=50, step=1, description='Perplexity 1:')
    perplexity_slider_2 = widgets.IntSlider(value=5, min=1, max=50, step=1, description='Perplexity 2:')

    plot = widgets.interactive(update_plot, perplexity_1=perplexity_slider_1, perplexity_2=perplexity_slider_2)
    display(plot)

def get_group_and_exp_size(codedict_path):
    """
    Get the group size (number of recordings) and the number of expressions in the codedict directory.

    Parameters:
    - codedict_path (str): The path to the codedict directory.

    Returns:
    - tuple: A tuple containing group size (int) and number of expressions (int).
    """

    # Calculate group size (number of folders)
    group_size = len(os.listdir(codedict_path))

    # Find the subfolder with the most expressions
    subfolders = [f.path for f in os.scandir(codedict_path) if f.is_dir()]
    expressions_folder = max(subfolders, key=lambda x: len(os.listdir(x)))

    # Calculate the number of expressions in the identified subfolder
    expressions = len(os.listdir(expressions_folder))

    return group_size, expressions

def get_params(codedict_path, exp_size=50, pose_size=6, shape_size=100, add_path=""):
    """
    Extract expression, pose, and shape parameters from codedict files.

    Parameters:
    - codedict_path (str): The path to the codedict directory.
    - exp_size (int): The size of the expression parameter array.
    - pose_size (int): The size of the pose parameter array.
    - shape_size (int): The size of the shape parameter array.

    Returns:
    - list: A list containing expression, pose, and shape parameter arrays.
    """

    # Get the Recording and Expressions numbers
    group_size, expressions = get_group_and_exp_size(codedict_path)
    print(f"Anzahl Aufnahmen: {group_size}")
    print(f"Anzahl Max Expressionen: {expressions}")

    # Initialize arrays to store parameters
    exp_params_left = np.zeros((expressions,group_size,exp_size))
    exp_params_right = np.zeros((expressions,group_size,exp_size))
    pose_params_left = np.zeros((expressions,group_size,pose_size))
    pose_params_right = np.zeros((expressions,group_size,pose_size))
    shape_params_left = np.zeros((expressions,group_size,shape_size))
    shape_params_right = np.zeros((expressions,group_size,shape_size))

    # Find the smallest number for ithe right index
    smallest_num = 100000

    for fp in os.listdir(codedict_path):
        pattern = r"_(\d+)$"
        match = re.search(pattern, fp)
        if match and pattern != "_7":
            num_rec = int(match.group(1))
            smallest_num = min(smallest_num, num_rec)
    
    print(smallest_num)

    # Extract parameters from codedict files
    for fp in os.listdir(codedict_path):
        pattern = r"_(\d+)$"
        match = re.search(pattern, fp)
        if match:
            num_rec = int(match.group(1))
            for fp_exp in os.listdir(os.path.join(codedict_path,fp)):
                pattern = r"_0?(\d+)$"
                match = re.search(pattern, fp_exp)
                if match:
                    num_exp = int(match.group(1))
                    codedict = torch.load(os.path.join(codedict_path,fp,fp_exp,fp_exp+f"{add_path}_codedict.pth"))
                    exp_params_left[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["left"]["exp"].cpu())[:exp_size]
                    exp_params_right[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["right"]["exp"].cpu())[:exp_size]
                    pose_params_left[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["left"]["pose"].cpu())[:pose_size]
                    pose_params_right[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["right"]["pose"].cpu())[:pose_size]
                    shape_params_left[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["left"]["shape"].cpu())[:shape_size]
                    shape_params_right[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["right"]["shape"].cpu())[:shape_size]

    return [exp_params_left, exp_params_right, pose_params_left, pose_params_right, shape_params_left, shape_params_right]

def run_deca(image_path:str, results_path=None, only_encode=True, wo_shape=True, pretrained_modelpath=None):
    """
    Run DECA to process an input image with different configuration options
    Also for run mode with a trained model possible.
    Default DECA: only_encode=False, wo_shape=False, pretrained_modelpath=None (None=using DECA Model)

    Args:
        image_path (str): Path to the input image.
        results_path (str, optional): Path to save the results of encoding (and decoding)
        only_encode (bool, optional): If True, only perform encoding; if False, perform full DECA processing with FLAME Model decoding(default is True).
        wo_shape (bool, optional): If True, exclude shape information; if False, include shape information in encoding (default is True).
        pretrained_modelpath (str, optional): Path to the pretrained DECA model checkpoint.

    Returns:
        dict: A dictionary containing DECA output, such as parameters and visualizations.
    """
    # Pretrained Model configuration only when without shape
    if pretrained_modelpath is not None and wo_shape:
        deca_cfg.pretrained_modelpath = pretrained_modelpath
    
    # Initialization of DECA
    deca = DECA(deca_cfg, only_encode=only_encode, wo_shape=wo_shape)

    # Get the Shape Parameters for decoding, if needed and the model is without shape
    if wo_shape and not only_encode:
        params = run_deca(image_path, results_path, wo_shape=False)
        shape_params = params['shape']
    else:
        shape_params=None
    #Get the data in the right format
    test = datasets.TestData(image_path)
    render_orig = True
    #Get the codedictionarys
    name = test[0]['imagename']
    images = test[0]['image'].to('cuda')[None,...]
    # Encode and optional decoding
    with torch.no_grad(): 
        codedict = deca.encode(images)
        if not only_encode:
            opdict, visdict = deca.decode(codedict, shape_params=shape_params) #tensor
            if render_orig:
                tform = test[0]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to('cuda')
                original_image = test[0]['original_image'][None, ...].to('cuda')
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform, shape_params=shape_params)    
                orig_visdict['inputs'] = original_image    
            if results_path != None:
                deca.save_obj(os.path.join(results_path, name + '.obj'), opdict)
                cv2.imwrite(os.path.join(results_path, name + '_vis.png'), deca.visualize(visdict))

    # Save codedictionary
    
        torch.save(codedict, os.path.join(results_path, name + '_codedict.pth'))

    torch.cuda.empty_cache()

    return codedict

def blendshapes_for_dataset(base_folder, num_exp=9, num_bs=52, end = "", pretrained_modelpath="/home/dietrich/Testing/DECA/DECA/data/deca_model.tar"):
    entries = os.listdir(base_folder)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(base_folder, entry))]
    individuals = len(folders)
    bs_list = np.zeros((individuals, num_exp, num_bs))
    count = 0
    for folder_name in os.listdir(base_folder):
        print(folder_name)
        folder_path = os.path.join(base_folder, folder_name)
        for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.lower().endswith((f'{end}.jpg', f'{end}.JPG')):
                    match = re.search(r'\d+', file_name[::-1])
                    exp_num = int(match.group()[::-1])
                    if exp_num == 11:
                        exp_num = 9
                    if not exp_num >= 10:
                        blendshapes = run_deca(file_path, wo_shape=True, pretrained_modelpath=pretrained_modelpath)
                        bs_list[count, exp_num-1] = blendshapes['exp'].cpu().tolist()[0]
        count+=1
    
    return bs_list

def show_tSNE(blendshapes, expressionnames, position="upper right"):
    expression_colors = {
    "Neutral": "cyan",
    "Stirn runzeln": "orange",
    "Augen zu": "lightgreen",
    "Augen doll zu": "green",
    "Nase rümpfen": "purple",
    "Lächeln": "mistyrose",
    "Lächeln doll": "red",
    "Mund spitzen": "gray",
    "Wangen aufblasen":"yellow",
    "Zähne fletschen":"brown",
    "Mundwinkel hoch":"blue",
    "Mundwinkel runter": "lightblue",
    }
    # Reshape bs_list zu (234, 52), wo 234 die Anzahl der Datenpunkte ist (26 * 9)
    bs_list_reshaped = blendshapes.reshape(-1, blendshapes.shape[-1])

    # Erstelle ein Label-Array für die Expression-Namen
    labels = np.array([expressionnames[i % len(expressionnames)] for i in range(bs_list_reshaped.shape[0])])

    # Wende t-SNE an
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(bs_list_reshaped)

    # Zähle die Anzahl der Punkte pro Expression
    point_counts = {label: np.sum(labels == label) for label in expressionnames}

    # Sortiere die Expressionen nach der Anzahl der Punkte
    sorted_expressionnames = sorted(expressionnames, key=lambda x: point_counts[x])

    # Plotte die Daten mit verschiedenen Farben für verschiedene Expression-Namen
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=expression_colors[name], markersize=10)
                    for name in sorted_expressionnames]

    plt.figure(figsize=(12, 8))
    for label in sorted_expressionnames:
        indices = np.where(labels == label)
        plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], c=expression_colors[label], s=50)

    plt.legend(handles=legend_elements, loc=position)
    plt.title('t-SNE Visualization of Blendshapes with Expression Colors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
