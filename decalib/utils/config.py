'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda'
cfg.device_id = '0'

cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
cfg.rasterizer_type = 'pytorch3d'
cfg.train_mode = '' # For training without shape: 'without_shape'
# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
# Paths to models to use
cfg.model.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = os.path.join(cfg.deca_dir, 'data', 'texture_data_256.npy')
cfg.model.fixed_displacement_path = os.path.join(cfg.deca_dir, 'data', 'fixed_displacement_256.npy')
cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl') 
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy') 
cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png') 
cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png') 
cfg.model.mean_tex_path = os.path.join(cfg.deca_dir, 'data', 'mean_texture.jpg') 
cfg.model.tex_path = os.path.join(cfg.deca_dir,  'data', 'FLAME_albedo_from_BFM.npz') 
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.uv_size = 256 # Mapping size from 2D -> 3D
cfg.model.param_list_wo_shape = ['tex', 'exp', 'pose', 'cam', 'light'] # Parameters that are calculatet
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100 # number of shape parameters used, can be 100-300 (more than 100 has no significant effect)
cfg.model.n_tex = 50 # number of texture parameters used
cfg.model.n_exp = 50 # number of expression parameters used, can be 50-100 (more than 50 has no significant effect)
cfg.model.n_cam = 3 # number of cam parameters
cfg.model.n_pose = 6 # number of pose parameters 
cfg.model.n_light = 27 # number of light parameters
cfg.model.use_tex = False # before: True, we do not need texture
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
# face recognition model
cfg.model.fr_model_path = os.path.join(cfg.deca_dir, 'data', 'resnet50_ft_weight.pkl')

## details
cfg.model.n_detail = 128 # number of detail parameters
cfg.model.max_z = 0.01 # 

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['vggface2hq'] # training dataset set
cfg.dataset.eval_data = ['vggface2hq'] # evaluation dataset
cfg.dataset.test_data = [''] # test dataset
cfg.dataset.batch_size = 8 # Number of Samples before updating parameters and evaluate
cfg.dataset.K = 6 # Number of Images per Sample! used for training; need to be less than 6
cfg.dataset.isSingle = False # If a Sample got only one image
cfg.dataset.num_workers = 2 # number of subprocesses used for data loading
cfg.dataset.image_size = 224 # size of square images: image_sizex  image_size
cfg.dataset.scale_min = 1.4 # minimale scale of images
cfg.dataset.scale_max = 1.8 # maximum scale of images
cfg.dataset.trans_scale = 0. #transformation = 0

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.train_detail = False #  False: Coarse mode, True: Detail mode
cfg.train.max_epochs = 50 # Number of Epochs; Number of passes through whole training dataset
cfg.train.max_steps = cfg.train.max_epochs*10000 # Number of iterations thtough optimization algorithm
cfg.train.lr = 1e-4 # learning rate; size of step during optimization
cfg.train.log_dir = 'logs' # directory for logs
cfg.train.log_steps = 10 # Based on number of steps: How often logging
cfg.train.vis_dir = 'train_images' # visualization of training images
cfg.train.vis_steps = 200 # How often saving after number of steps
cfg.train.write_summary = True 
cfg.train.checkpoint_steps = 500 # How often saving models state after x number of steps
cfg.train.val_steps = 1000 # How often validation after num of steps
cfg.train.val_vis_dir = 'val_images'
cfg.train.eval_steps = 5000 # How often evaluation after number of steps
cfg.train.resume = False # Resume training after last checkpoint

# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.lmk = 1.0 # Landmark loss: !Absolute! Difference between ground truth landmarks and FLAME Landmarks
cfg.loss.useWlmk = True # Use weighted Landmark loss
cfg.loss.eyed = 1.0 # !Relative! Offset between upper and lower eyelid landmarks
cfg.loss.lipd = 0.5 # Lip region landmark loss? No more description found
cfg.loss.photo = 2.0 # Error between Input Image and Rendering via Hadamar Product
cfg.loss.useSeg = False # Changed, because we do not have the required mask
cfg.loss.id = 0.0 #not used, cause we do not train identity
cfg.loss.id_shape_only = False
cfg.loss.reg_exp = 1e-04 # Weight for the expression regularization loss
cfg.loss.reg_tex = 1e-04 # Weight for the Texture regularization loss
cfg.loss.reg_light = 1.0 # Weight for the regularization loss on lighting parameters.
cfg.loss.reg_jaw_pose = 0.0 #1. Weight for the regularization loss on jaw pose. If set to a non-zero value, it enforces regularization on jaw pose.
cfg.loss.use_gender_prior = False # Bool to se Gender information
cfg.loss.shape_consistency = False
# loss for detail
cfg.loss.detail_consistency = True # Disentaglement between identity and expression dependent details
cfg.loss.useConstraint = True # 
cfg.loss.mrf = 5e-2 # Reconstructing geometric details: Implicit Diversified Markov Random Field (ID-MRF) loss
cfg.loss.photo_D = 2.0 # Weight for the photometric loss on the detail
cfg.loss.reg_sym = 0.005 # add robustness to self-occlusions; soft symmetry loss
cfg.loss.reg_z = 0.005 # something with uv map 
cfg.loss.reg_diff = 0.005 # shading smootheness

cfg.output_dir = ''

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default = 'train', help='deca mode')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.mode
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
