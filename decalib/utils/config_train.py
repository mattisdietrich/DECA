'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

config_train = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_train.deca_dir = abs_deca_dir
config_train.device = 'cuda'
config_train.device_id = '0'

config_train.pretrained_modelpath = '' #os.path.join(config_train.deca_dir, 'data', 'deca_model.tar')
config_train.output_dir = ''
config_train.rasterizer_type = 'pytorch3d'
# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
config_train.model = CN()
# Paths to models to use
config_train.model.topology_path = os.path.join(config_train.deca_dir, 'data', 'head_template.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
config_train.model.dense_template_path = os.path.join(config_train.deca_dir, 'data', 'texture_data_256.npy')
config_train.model.fixed_displacement_path = os.path.join(config_train.deca_dir, 'data', 'fixed_displacement_256.npy')
config_train.model.flame_model_path = os.path.join(config_train.deca_dir, 'data', 'generic_model.pkl') 
config_train.model.flame_lmk_embedding_path = os.path.join(config_train.deca_dir, 'data', 'landmark_embedding.npy') 
config_train.model.face_mask_path = os.path.join(config_train.deca_dir, 'data', 'uv_face_mask.png') 
config_train.model.face_eye_mask_path = os.path.join(config_train.deca_dir, 'data', 'uv_face_eye_mask.png') 
config_train.model.mean_tex_path = os.path.join(config_train.deca_dir, 'data', 'mean_texture.jpg') 
config_train.model.tex_path = os.path.join(config_train.deca_dir,  'data', 'FLAME_albedo_from_BFM.npz') 
config_train.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
config_train.model.uv_size = 256 # Mapping size from 2D -> 3D
config_train.model.param_list = ['tex', 'exp', 'pose', 'cam', 'light'] # Parameters that are calculatet
config_train.model.n_shape = 100 # number of shape parameters used, can be 100-300 (more than 100 has no significant effect)
config_train.model.n_tex = 50 # number of texture parameters used
config_train.model.n_exp = 50 # number of expression parameters used, can be 50-100 (more than 50 has no significant effect)
config_train.model.n_cam = 3 # number of cam parameters
config_train.model.n_pose = 6 # number of pose parameters 
config_train.model.n_light = 27 # number of light parameters
config_train.model.use_tex = False # before: True, wen do not need texture
config_train.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
# face recognition model
config_train.model.fr_model_path = os.path.join(config_train.deca_dir, 'data', 'resnet50_ft_weight.pkl')

## details
config_train.model.n_detail = 128 # number of detail parameters
config_train.model.max_z = 0.01 # 

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
config_train.dataset = CN()
config_train.dataset.training_data = ['vggface2hq'] # training dataset set
config_train.dataset.eval_data = ['vggface2hq'] # evaluation dataset
config_train.dataset.test_data = [''] # test dataset
config_train.dataset.batch_size = 6 # Number of Samples before updating parameters and evaluate
config_train.dataset.K = 4 # Number of Images per Sample! used for training; need to be less than 6
config_train.dataset.isSingle = False # If a Sample got only one image
config_train.dataset.num_workers = 2 # number of subprocesses used for data loading
config_train.dataset.image_size = 224 # size of square images: image_sizex  image_size
config_train.dataset.scale_min = 1.4 # minimale scale of images
config_train.dataset.scale_max = 1.8 # maximum scale of images
config_train.dataset.trans_scale = 0. #transformation = 0

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
config_train.train = CN()
config_train.train.train_detail = False #  False: Coarse mode, True: Detail mode
config_train.train.max_epochs = 10 # Number of Epochs; Number of passes through whole training dataset
config_train.train.max_steps = 1000000 # Number of iterations thtough optimization algorithm
config_train.train.lr = 1e-4 # learning rate; size of step during optimization
config_train.train.log_dir = 'logs' # directory for logs
config_train.train.log_steps = 10 # Based on number of steps: How often logging
config_train.train.vis_dir = 'train_images' # visualization of training images
config_train.train.vis_steps = 200 # How often saving after number of steps
config_train.train.write_summary = True 
config_train.train.checkpoint_steps = 500 # How often saving models state after x number of steps
config_train.train.val_steps = 500 # How often validation after num of steps
config_train.train.val_vis_dir = 'val_images'
config_train.train.eval_steps = 5000 # How often evaluation after number of steps
config_train.train.resume = False # Resume training after last checkpoint

# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
config_train.loss = CN()
config_train.loss.lmk = 1.0 # Landmark loss: !Absolute! Difference between ground truth landmarks and FLAME Landmarks
config_train.loss.useWlmk = True # Use weighted Landmark loss
config_train.loss.eyed = 1.0 # !Relative! Offset between upper and lower eyelid landmarks
config_train.loss.lipd = 0.5 # Lip region landmark loss? No more description found
config_train.loss.photo = 2.0 # Error between Input Image and Rendering via Hadamar Product
config_train.loss.useSeg = False # Changed, because we do not have the required mask
#config_train.loss.id = 0.2 not used, cause we do not train identity
#config_train.loss.id_shape_only = False
config_train.loss.reg_exp = 1e-04 # Weight for the expression regularization loss
config_train.loss.reg_tex = 1e-04 # Weight for the Texture regularization loss
config_train.loss.reg_light = 1. # Weight for the regularization loss on lighting parameters.
config_train.loss.reg_jaw_pose = 0. #1.Weight for the regularization loss on jaw pose. If set to a non-zero value, it enforces regularization on jaw pose.
config_train.loss.use_gender_prior = False # Bol to se Gender information
#config_train.loss.shape_consistency = False
# loss for detail
config_train.loss.detail_consistency = True # Disentaglement between identity and expression dependent details
config_train.loss.useConstraint = True # 
config_train.loss.mrf = 5e-2 # Reconstructing geometric details: Implicit Diversified Markov Random Field (ID-MRF) loss
config_train.loss.photo_D = 2. # Weight for the photometric loss on the detail
config_train.loss.reg_sym = 0.005 # add robustness to self-occlusions; soft symmetry loss
config_train.loss.reg_z = 0.005 # somethign with uv map 
config_train.loss.reg_diff = 0.005 # shading smootheness


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return config_train.clone()

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
