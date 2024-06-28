import sys
import os
import argparse
import torch

sys.path.append("./Core")
from model import TestModel
from dataset import *
from cross_validation import cross_validation
from utils import Init_random_seed, clearOldLogs
from default_configs import *

parser = argparse.ArgumentParser()
parser.add_argument('exp',
                    help='name of experiment')
parser.add_argument('--dataset', '-dataset', type=str, default="gowalla",
                    help='dataset on which experiment is conducted')
parser.add_argument('--batch_size', '-bs', type=int, default=8,
                    help='batch size for one iteration during training')
parser.add_argument('--lr', '-lr', type=float, default=1e-5,
                    help='learning rate parameter')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5,
                    help='weight decay parameter')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability)')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu')
parser.add_argument('--max_epoch', '-max_epoch', type=int, default=200,
                    help='maximal training epochs')
parser.add_argument('--lambda_tradeoff', '-lambda', type=float, default=1.0,
                    help='trade-off parameter for embedding loss')
parser.add_argument('--class_emb', '-class_emb', type=int, default=256,
                    help='dimensionality of label embedding')
parser.add_argument('--in_layers', '-in_layers', type=int, default=1,
                    help='number of layers for obtaining latent representation')
parser.add_argument('--cuda', '-cuda', action='store_true',
                    help='whether to use gpu')
parser.add_argument('--quiet', '-quiet', action='store_true',
                    help='whether to train in quiet mode')
parser.add_argument('--ada_epoch', '-ada_epoch', action='store_true',
                    help='whether to decide the max_epoch on the first fold during cross-validation')
parser.add_argument('--topk', '-topk', type=int, default=4,
                    help='choose top k to recommand')
parser.add_argument('--default_cfg', '-default_cfg', action='store_true',
                    help='whether to run experiment with default hyperparameters')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Setting random seeds
    Init_random_seed()
    
    # Loading dataset
    dataset_name = args.dataset
    print(dataset_name)
    dataset = eval(dataset_name)()
    X, y, X_train, X_test, y_train, y_test, label_adj = dataset.Data()
    
    # Setting configurations
    configs = generate_default_config()
    configs['dataset'] = dataset
    configs['dataset_name'] = dataset.Name()
    
    configs['in_features'] = X.size(1)
    configs['num_classes'] = y.size(1)
    configs['in_num'] = X_train.size(0)
    configs['class_emb'] = args.class_emb
    configs['in_layers'] = args.in_layers
    
    configs['lambda'] = args.lambda_tradeoff
    configs['lr'] = args.lr
    configs['weight_decay'] = args.weight_decay
    configs['hidden'] = args.hidden
    configs['dropout'] = args.dropout
    configs['alpha'] = args.alpha
    configs['batch_size'] = args.batch_size
    configs['max_epoch'] = args.max_epoch
    configs['ada_epoch'] = args.ada_epoch
    configs['use_gpu'] = args.cuda
    configs['device'] = torch.device('cuda' if torch.cuda.is_available() and configs['use_gpu'] else 'cpu')
    configs['topk'] = args.topk
    
    configs['exp'] = args.exp
    configs['model_name'] = 'TestModel'
    configs['exp_dir'] = os.path.join(configs['model_name'],
                                      configs['exp'],
                                      configs['dataset_name'])
    configs['save_checkpoint_path'] = os.path.join(configs['exp_dir'], 'checkpoint')
    
    # Clear old logs
    clearOldLogs(os.path.join(configs['model_name'], configs['exp']))
    
    # Loading dataset-specific configs
    if args.default_cfg:
        eval('{}_configs'.format(configs['dataset_name']))(configs)
    
    # Creating model
    model = TestModel(configs)
    
    # Cross-validation
    val_metrics, _ = cross_validation(model, dataset, [X_train, X_test, y_train, y_test, label_adj], random_state=configs['rand_seed'],
                                      quiet_mode=args.quiet, ada_epoch=configs['ada_epoch'],
                                      save_model=True)
    
    # Displaying results of cross-validation
    for key in val_metrics:
        print('{}: {:.4f}'.format(key, val_metrics[key].value()[0]))
    