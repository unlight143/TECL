import torch

def generate_default_config():
    configs = {}
    
    # Device
    configs['use_gpu'] = torch.cuda.is_available()
    configs['use_multi_gpu'] = configs['use_gpu'] and torch.cuda.device_count() > 1
    configs['device'] = torch.device('cuda' if torch.cuda.is_available() 
                                     and configs['use_gpu'] else 'cpu')
    
    # Dataset
    configs['dataset'] = None
    
    # Training parameters
    configs['dtype'] = torch.float
    configs['lr'] = 1e-3
    configs['weight_decay'] = 1e-5
    configs['batch_size'] = 1000
    configs['start_epoch'] = 0
    configs['max_epoch'] = 200
    configs['evaluate'] = False
    configs['pre_sigmoid_bias'] = 0.5
    
    # Training information display and log
    configs['display'] = True
    configs['display_freq'] = 10
    configs['save_checkpoint_path'] = 'checkpoint'
    configs['exp'] = 'exp'
    
    # Reproducibility
    configs['rand_seed'] = 0
    
    return configs

def CAL500_configs(config):
    config['class_emb'] = 64
    config['lambda'] = 1e-4
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.4
    config['ada_epoch'] = True
    
def Image_configs(config):
    config['class_emb'] = 64
    config['lambda'] = 0
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.4
    config['ada_epoch'] = True

def scene_configs(config):
    config['class_emb'] = 128
    config['lambda'] = 1e-3
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.4
    config['ada_epoch'] = True
    
def yeast_configs(config):
    config['class_emb'] = 256
    config['lambda'] = 1e-3
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.4
    config['ada_epoch'] = True
    
def corel5k_configs(config):
    config['class_emb'] = 256
    config['lambda'] = 1e-4
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.3
    config['ada_epoch'] = True
    
def rcv1subset1_configs(config):
    config['class_emb'] = 256
    config['lambda'] = 1e-1
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.1
    config['ada_epoch'] = True

def Corel16k001_configs(config):
    config['class_emb'] = 256
    config['lambda'] = 0
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.2
    config['ada_epoch'] = True
    
def delicious_configs(config):
    config['class_emb'] = 256
    config['lambda'] = 1e-1
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.4
    config['ada_epoch'] = True
    
def iaprtc12_configs(config):
    config['class_emb'] = 64
    config['lambda'] = 1
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.2
    config['ada_epoch'] = True
    
def espgame_configs(config):
    config['class_emb'] = 128
    config['lambda'] = 1e-5
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.2
    config['ada_epoch'] = True
    
def mirflickr_configs(config):
    config['class_emb'] = 256
    config['lambda'] = 1e-2
    config['in_layers'] = 2
    config['pre_sigmoid_bias'] = 0.3
    config['ada_epoch'] = True
    
def tmc2007_configs(config):
    config['class_emb'] = 256
    config['lambda'] = 1
    config['in_layers'] = 2
    config['pre_sigmoid_bias'] = 0.3
    config['ada_epoch'] = True
    
def mediamill_configs(config):
    config['class_emb'] = 256
    config['lambda'] = 1e-4
    config['in_layers'] = 2
    config['pre_sigmoid_bias'] = 0.3
    config['ada_epoch'] = True

def bookmarks_configs(config):
    config['class_emb'] = 128
    config['lambda'] = 1e-2
    config['in_layers'] = 1
    config['pre_sigmoid_bias'] = 0.2
    config['ada_epoch'] = True