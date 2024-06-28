import torch
import torch.nn as nn
import math
import os

from layers import GAT, FDModel, GIN, AutoEncoder, VGAEModel
from engine import CLIFModelEngine
from loss import LinkPredictionLoss_cosine
from utils import Init_random_seed, compute_precision, compute_recall, compute_mrr, compute_rmse
from metrics import *

class TestNet(nn.Module):
    def __init__(self, configs):
        super(TestNet, self).__init__()
        self.rand_seed = configs['rand_seed']
        
        # Label semantic encoding module
        self.label_embedding = nn.Parameter(torch.eye(configs['in_num']),
                                            requires_grad=False)
        self.label_adj = nn.Parameter(torch.eye(configs['in_num']),
                                      requires_grad=False)
        self.GAT_layer = GAT(configs['in_num'], configs['class_emb'], configs['class_emb'], 
                                configs['dropout'], configs['alpha'], 8)
        # self.GAT_layer = GIN(2, configs['num_classes'], configs['class_emb'],
        #                        [math.ceil(configs['class_emb'] / 2)])
        # self.ae_layer = AutoEncoder(configs['num_classes'], configs['class_emb'])
        # self.Vage_layer = VGAEModel(configs['num_classes'], int(2*configs['class_emb']), configs['class_emb'])
        
        # Semantic-guided feature-disentangling module
        self.FD_model = FDModel(configs['in_features'], configs['class_emb'],
                                512, 512, configs['in_layers'], 1,
                                False, 'leaky_relu', 0.1)
        
        # Classifier
        self.cls_conv = nn.Conv1d(configs['in_num'], configs['num_classes'],
                                  512, groups=1)
        
        # Moving model to the right device for consistent initialization
        self.to(configs['device'])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_embedding)
        self.GAT_layer.reset_parameters()
        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()
        
    def get_config_optim(self):
        return [{'params': self.GAT_layer.parameters()},
                {'params': self.FD_model.parameters()},
                {'params': self.cls_conv.parameters()}]

    def forward(self, input):
        # Generating semantic label embeddings via label semantic encoding module
        label_embedding = self.GAT_layer(self.label_embedding, self.label_adj)
        # label_embedding = self.Vage_layer(self.label_embedding, self.label_adj)
        
        # Generating label-specific features via semantic-guided feature-disentangling module
        X = self.FD_model(input, label_embedding)
        
        # Classification
        output = self.cls_conv(X).squeeze(2)
        
        return output, label_embedding

class TestModel(nn.Module):
    def __init__(self, configs):
        super(TestModel, self).__init__()
        self.configs = configs
        
        if self._configs('label_metrics') is None:
            self.configs['label_metrics'] = ["HammingLoss", "AdjustedHammingLoss"]
        if self._configs('score_metrics') is None:
            self.configs['score_metrics'] = ["OneError", "Coverage", "RankingLoss",
                                             "AveragePrecision", "MacroAUC"]
        if self._configs('dtype') is None:
            self.configs['dtype'] = torch.float
        
        # Creating model
        self.model = TestNet(self.configs)
        
        # Defining loss function
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.emb_criterion = LinkPredictionLoss_cosine()
        
        # Creating learning engine
        self.engine = CLIFModelEngine(self.configs)
        
    def reset_parameters(self):
        Init_random_seed(self.configs['rand_seed'])
        self.model.reset_parameters()
        
    def _configs(self, name):
        if name in self.configs:
            return self.configs[name]
    
    def Train(self, X_train, y_train, adj_data, X_val=None, y_val=None, quiet_mode=False):
        '''
        Parameters
        ----------
        X_train : Tensor
            MxN Tensor, the ith training instance is stored in X_train[i,:].
        y_train : Tensor
            MxQ Tensor, if the ith training instance belongs to the jth class,
            then y_train[i,j] equals to +1, otherwise y_train[i,j] equals to 0.
        adj_data : Tensor
            QxQ Tensor, label adjacency matrix
        X_val : Tensor
            MxN Tensor, the ith validation instance is stored in X_train[i,:].
        y_val : Tensor
            MxQ Tensor, if the ith validation instance belongs to the jth class,
            then y_val[i,j] equals to +1, otherwise y_val[i,j] equals to 0.

        '''
        X_train, y_train, X_val, y_val = self.on_start_train(X_train, y_train,
                                                             X_val, y_val)
        self.model.label_adj.data = adj_data
        
        # Learning
        self.engine.Learn(self.model, [self.criterion, self.emb_criterion], X_train,
                          y_train, X_val, y_val, quiet_mode)
        
    def on_start_train(self, X_train, y_train, X_val=None, y_val=None):
        # Casting all parameters of the model to the specific dtype
        self.dtype()
        X_train = X_train.type(self.configs['dtype'])
        y_train = y_train.type(self.configs['dtype'])
        if X_val is not None:
            X_val = X_val.type(self.configs['dtype'])
        if y_val is not None:
            y_val = y_val.type(self.configs['dtype'])
        
        # Generating the adjacency matrix for the label semantic encoding module
        # self.model.label_adj.data = self.sym_conditional_prob(y_train)
        
        return X_train, y_train, X_val, y_val
    
    def sym_conditional_prob(self, y):
        adj = torch.matmul(y.t(), y)
        y_sum = torch.sum(y.t(), dim=1, keepdim=True)
        y_sum[y_sum<1e-6] = 1e-6
        adj = adj / y_sum
        for i in range(adj.size(0)):
            adj[i,i] = 0.0
        adj = (adj + adj.t()) * 0.5
        return adj
    
    def Predict(self, X):
        '''
        Parameters
        ----------
        X : Tensor
            MxN Tensor, the ith instance is stored in X[i,:].

        Returns
        -------
        pred_labels : Tensor
            MxQ Tensor, if the ith instance belongs to the jth class, then
            pred_labels[i,j] equals to +1, otherwise pred_labels[i,j] equals to 0.
        pred_probs : Tensor
            MxQ Tensor, the probability of the ith instance belonging to the 
            jth class is stored in pred_probs[i,j]
        '''
        self.model.eval()
        batch_size = 2 * self.configs['batch_size']
        bias = math.log(self.configs['pre_sigmoid_bias']/(1-self.configs['pre_sigmoid_bias'])) # a bias term to calibrate predicted logits
        with torch.no_grad():
            X = self.on_start_predict(X)
            
            if batch_size >= X.size(0):
                X = X.to(self.configs['device'])
                output = self.model(X)[0] - bias
            else:
                # Splitting dataset into batches
                num_data = X.size(0)
                max_iters = math.ceil(num_data / batch_size)
                output = []
                for i in range(max_iters):
                    low = i*batch_size
                    high = min(low+batch_size, num_data)
                    X_batch = X[low:high, :].to(self.configs['device'])
                    output.append(self.model(X_batch)[0] - bias)
                output = torch.cat(output, dim=0)
            
            pred_probs = output.sigmoid_()
            pred_labels = pred_probs.new_zeros(pred_probs.size())
            pred_labels[pred_probs > 1e-3] = 1
        return pred_labels, pred_probs
    
    def on_start_predict(self, X):
        X = X.type(self.configs['dtype'])
        return X
    
    def Evaluate(self, pred_labels, pred_probs, y):
        '''
        Parameters
        ----------
        pred_labels : Tensor
            MxQ Tensor, if the ith instance belongs to the jth class, then
            pred_labels[i,j] equals to +1, otherwise pred_labels[i,j] equals to 0.
        pred_probs : Tensor
            MxQ Tensor, the probability of the ith instance belonging to the 
            jth class is stored in pred_probs[i,j]
        y : Tensor
            MxQ Tensor, if the ith instance belongs to the jth class, then
            y[i,j] equals to +1, otherwise y[i,j] equals to 0.

        Returns
        -------
        metrics : dict
            Metrics for evaluation.
        '''
        pred_labels = pred_labels.to(y.device)
        pred_probs = pred_probs.to(y.device)

        metrics = {}
        for metric_name in self.configs['label_metrics']:
            metrics[metric_name] = eval(metric_name)(pred_labels, y)
        for metric_name in self.configs['score_metrics']:
            metrics[metric_name] = eval(metric_name)(pred_probs, y)
        metrics["Precision"] = compute_precision(pred_probs, y, self.configs['topk'])
        metrics["Recall"] = compute_recall(pred_probs, y, self.configs['topk'])
        metrics['Mrr'] = compute_mrr(pred_probs, y)
        metrics["Rmse"] = compute_rmse(pred_probs, y, self.configs['topk'])
        return metrics
    
    def Test(self, X, y):
        '''
        Parameters
        ----------
        X : Tensor
            MxN Tensor, the ith instance is stored in X[i,:].
        y : Tensor
            MxQ Tensor, if the ith instance belongs to the jth class, then
            y[i,j] equals to +1, otherwise y[i,j] equals to 0.

        Returns
        -------
        metrics : dict
            Metrics for evaluation.
        '''
        pred_labels, pred_probs = self.Predict(X)
        return self.Evaluate(pred_labels, pred_probs, y)
    
    def Load_checkpoint(self, checkpoint):
        if os.path.isfile(checkpoint):
            checkpoint = torch.load(checkpoint)
            self.configs['start_epoch'] = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.configs['dtype'] = list(self.model.parameters())[0].dtype
        else:
            print('No checkpoing is found at {}.'.format(checkpoint))
    
    def dtype(self, dtype=None):
        '''
        Changing the dtype of parameters in the model
        '''
        if dtype is not None:
            self.configs['dtype'] = dtype
        self.model.type(self.configs['dtype'])
