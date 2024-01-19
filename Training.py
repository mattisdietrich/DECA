from decalib.deca import DECA
from decalib.trainer import Trainer
from decalib.utils.config_train import config_train as cfg
import yaml
import shutil
import torch.backends.cudnn as cudnn
import os
import torch
import warnings


def run():
    warnings.simplefilter("ignore")
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    #with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
    #    yaml.dump(cfg, f, default_flow_style=False)
    #shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))
        
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # start training
    # deca model
    cfg.rasterizer_type = 'pytorch3d'
    deca = DECA(cfg)
    trainer = Trainer(model=deca, config=cfg)

    
    ## start train
    trainer.fit()

run()