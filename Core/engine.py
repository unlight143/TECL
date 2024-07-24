import torch
import torchnet
import os
import math

from metrics import AveragePrecision, HammingLoss

class Engine():
    '''
    Engine to optimize a model
    '''
    def __init__(self, configs={}):
        self.configs = configs
        # Checking configurations
        if self.configs['use_gpu'] and not torch.cuda.is_available():
            self.configs['use_gpu'] = False
        self.configs['use_multi_gpu'] = self.configs['use_gpu'] and \
                                            torch.cuda.device_count() > 1
        self.configs['device'] = torch.device('cuda' if torch.cuda.is_available()
                                              and self.configs['use_gpu'] else 'cpu')
        
        self.state = {}
        self.state['ave_loss'] = torchnet.meter.AverageValueMeter()
    
    def _configs(self, name):
        if name in self.configs:
            return self.configs[name]
        
    def Learn(self, model, criterion, X_train, y_train, X_val=None, y_val=None, quiet_mode=False):
        if X_val is None or y_val is None:
            self.configs['evaluate'] = False
            
        if self.configs['use_multi_gpu']:
            model = torch.nn.DataParallel(model)
        model.to(self.configs['device'])
            
        # Defining optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['lr'], weight_decay=self.configs['weight_decay'])
        
        if self.configs['evaluate']:
            self.init_eval()
        
        self.quiet_mode = quiet_mode
        if not quiet_mode:
            print('Training begins ......')
            
        for epoch in range(self.configs['start_epoch'], self.configs['max_epoch']):
            self.state['epoch'] = epoch
            self.state['lr'] = self.configs['lr']
            
            # Training for one epoch
            self.train(model, criterion, X_train, y_train, optimizer)
            
            # Evaluate on validation set
            val_metric = None
            if self.configs['evaluate']:
                with torch.no_grad():
                    val_metric = self.validate(model, criterion, X_val, y_val)
            
            # Saving checkpoint
            self.on_save_checkpoint(model, val_metric)
        
        if not quiet_mode:
            print('Training finishes.')
    
    def train(self, model, criterion, X_train, y_train, optimizer):
        model.train()
        
        self.on_start_epoch(True)
        
        if self.configs['batch_size'] >= X_train.size(0):
            self.state['max_iters'] = 1
            X_train = X_train.to(self.configs['device'])
            y_train = y_train.to(self.configs['device'])
            self.Step(True, model, criterion, X_train, y_train, optimizer)
        else:
            # Splitting dataset into batches
            with torch.no_grad():
                num_data = X_train.size(0)
                index = torch.randperm(num_data)
                batch_size = self.configs['batch_size']
                self.state['max_iters'] = math.ceil(num_data / batch_size)
            for i in range(self.state['max_iters']):
                self.state['iteration'] = i
                low = i*batch_size
                high = min(low+batch_size, num_data)
                X = X_train[index[low:high], :]
                y = y_train[index[low:high], :]
                X = X.to(self.configs['device'])
                y = y.to(self.configs['device'])
                    
                self.Step(True, model, criterion, X, y, optimizer)
                    
                # Displaying training information for one iteration
                self.on_end_batch(True)
        
        # Displaying training information for one epoch
        self.on_end_epoch(True)
        
    def validate(self, model, criterion, X_val, y_val):
        model.eval()
        
        self.on_start_epoch(False)
        
        if self.configs['batch_size'] >= X_val.size(0):
            self.state['max_iters'] = 1
            X_val = X_val.to(self.configs['device'])
            y_val = y_val.to(self.configs['device'])
            self.Step(False, model, criterion, X_val, y_val)
        else:
            # Splitting dataset into batches
            num_data = X_val.size(0)
            index = torch.randperm(num_data)
            batch_size = self.configs['batch_size']
            self.state['max_iters'] = math.ceil(num_data / batch_size)
            for i in range(self.state['max_iters']):
                self.state['iteration'] = i
                low = i*batch_size
                high = min(low+batch_size, num_data)
                X = X_val[index[low:high], :]
                y = y_val[index[low:high], :]
                X = X.to(self.configs['device'])
                y = y.to(self.configs['device'])
                    
                self.Step(False, model, criterion, X, y)
                    
                # Displaying validation information for one iteration
                self.on_end_batch(False)
        
        # Displaying validation information for one epoch
        val_metric = self.on_end_epoch(False)
        
        return val_metric
        
    def Step(self, training, model, criterion, X, y, optimizer=None):
        '''
        Forwarding model and optimizing model once during training.
        '''
        output = model(X)
        loss = criterion(output, y)
        
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.state['cur_loss'] = loss.data.item()
        self.state['ave_loss'].add(self.state['cur_loss'])
    
    def init_eval(self):
        self.state['best_val_metric'] = {'Loss': float('inf')}
        self.state['min_val_loss'] = float('inf')
        self.state['no_update_epoch'] = 0
        
    def on_start_epoch(self, training):
        self.state['ave_loss'].reset()
        
    def on_end_epoch(self, training):
        '''
        Displaying information after one epoch finishing.
        '''
        if self.configs['display'] and not self.quiet_mode:
            if training:
                print('Epoch: [{}]\t lr {:.2e}\t Average Loss {:.4f}'.
                      format(self.state['epoch'],
                             self.state['lr'],
                             self.state['ave_loss'].value()[0]))
            else:
                print('Test: \t Average Loss {:.4f}'.
                      format(self.state['ave_loss'].value()[0]))

        return {'Loss': self.state['ave_loss'].value()[0]}
    
    def on_end_batch(self, training):
        '''
        Displaying information after one batch.
        '''
        if self.configs['display'] and self.state['iteration'] % self.configs['display_freq'] == 0 and \
            not self.quiet_mode:
            if training:
                print('Epoch: [{}][{}/{}]\t Loss {:.4f} ({:.4f})'.
                      format(self.state['epoch'],
                             self.state['iteration'],
                             self.state['max_iters'],
                             self.state['cur_loss'],
                             self.state['ave_loss'].value()[0]))
            else:
                print('Test: [{}/{}]\t Loss {:.4f} ({:.4f})'.
                      format(self.state['iteration'],
                             self.state['max_iters'],
                             self.state['cur_loss'],
                             self.state['ave_loss'].value()[0]))
    
    def save_checkpoint(self, checkpoint, is_best=False):
        '''
        Saving checkpoint.
        '''
        if not os.path.exists(self.configs['save_checkpoint_path']):
            os.makedirs(self.configs['save_checkpoint_path'])
        
        if is_best:
            filename = self.configs['dataset_name'] + '_best_checkpoint.pth'
            filename = os.path.join(self.configs['save_checkpoint_path'], filename)
            self.configs['best_checkpoint_path'] = filename
        else:
            filename = self.configs['dataset_name'] + '_checkpoint_' + \
                       str(self.state['epoch']+1) + '.pth'
            filename = os.path.join(self.configs['save_checkpoint_path'], filename)
        if not self.quiet_mode:
            print('saving model to {}'.format(filename))
        torch.save(checkpoint, filename)
    
    def on_save_checkpoint(self, model, val_metric=None):
        '''
        Saving checkpoint.
        '''
        if self.configs['evaluate']:
            if self.compare_metric(val_metric, self.state['best_val_metric']):
                
                self.state['best_val_metric'] = val_metric
                self.configs['best_epoch'] = self.state['epoch']+1
                # Saving checkpoint
                self.save_checkpoint({'epoch': self.state['epoch']+1,
                                      'state_dict': model.module.state_dict() 
                                      if self.configs['use_multi_gpu'] 
                                      else model.state_dict()}, True)
        return False
    
    def compare_metric(self, m1, m2, eps=1e-5):
        return m1['Loss'] < m2['Loss']

class MultiLabelEngine(Engine):
    '''
    Engine to optimize a multi-label classifier
    '''
    def __init__(self, configs={}):
        self.configs = configs
        # Checking configurations
        if self.configs['use_gpu'] and not torch.cuda.is_available():
            self.configs['use_gpu'] = False
        self.configs['use_multi_gpu'] = self.configs['use_gpu'] and \
                                            torch.cuda.device_count() > 1
        self.configs['device'] = torch.device('cuda' if torch.cuda.is_available()
                                              and self.configs['use_gpu'] else 'cpu')
        
        self.state = {}
        self.state['ave_loss'] = torchnet.meter.AverageValueMeter()
        self.state['ap'] = torchnet.meter.AverageValueMeter()
        self.state['hammingloss'] = torchnet.meter.AverageValueMeter()
        
    def Step(self, training, model, criterion, X, y, optimizer=None):
        '''
        Forwarding model and optimizing model once during training.
        '''
        output = model(X)
        loss = criterion(output, y)
        
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            pred_probs = output.sigmoid_()
            pred_labels = pred_probs.new_zeros(pred_probs.size())
            pred_labels[pred_probs > 0.5] = 1
            ap = AveragePrecision(pred_probs, y)
            self.state['ap'].add(ap)
            hammingloss = HammingLoss(pred_labels, y)
            self.state['hammingloss'].add(hammingloss)
            
        self.state['cur_loss'] = loss.data.item()
        self.state['ave_loss'].add(self.state['cur_loss'])
    
    def init_eval(self):
        self.state['best_val_metric'] = {'Loss': float('inf'),
                                         'AveragePrecision': 0.0,
                                         'HammingLoss': float('inf')}
        self.state['min_val_loss'] = float('inf')
        self.state['no_update_epoch'] = 0
        
    def on_start_epoch(self, training):
        Engine.on_start_epoch(self, training)
        self.state['ap'].reset()
        self.state['hammingloss'].reset()
    
    def on_end_epoch(self, training):
        '''
        Displaying information after one epoch finishing.
        '''
        if self.configs['display'] and not self.quiet_mode:
            if training:
                print('Epoch: [{}]\t lr {:.2e}\t Average Loss {:.4f}'.
                      format(self.state['epoch'],
                             self.state['lr'],
                             self.state['ave_loss'].value()[0]))
            else:
                print('Test: \t Average Loss {:.4f}\t AP {:.4f}\t HammingLoss {:.4f}'.
                      format(self.state['ave_loss'].value()[0],
                             self.state['ap'].value()[0],
                             self.state['hammingloss'].value()[0]))
        
        return {'Loss': self.state['ave_loss'].value()[0],
                'AveragePrecision': self.state['ap'].value()[0],
                'HammingLoss': self.state['hammingloss'].value()[0]}
    
    def compare_metric(self, m1, m2, eps=1e-5):
        if m1['AveragePrecision'] > m2['AveragePrecision']:
            return True
        
        if math.fabs(m1['AveragePrecision'] - m2['AveragePrecision']) < eps:
            if m1['HammingLoss'] < m2['HammingLoss']:
                return True
            
            if math.fabs(m1['HammingLoss'] - m2['HammingLoss']) < eps:
                return m1['Loss'] < m2['Loss']
        return False
            
class CLIFModelEngine(MultiLabelEngine):
    '''
    Engine to optimize CLIFModel.
    '''
    def __init__(self, configs={}):
        self.configs = configs
        # Checking configurations
        if self.configs['use_gpu'] and not torch.cuda.is_available():
            self.configs['use_gpu'] = False
        self.configs['use_multi_gpu'] = self.configs['use_gpu'] and \
                                            torch.cuda.device_count() > 1
        self.configs['device'] = torch.device('cuda' if torch.cuda.is_available()
                                              and self.configs['use_gpu'] else 'cpu')
        
        self.state = {}
        self.state['ave_loss'] = torchnet.meter.AverageValueMeter()
        self.state['ave_cls_loss'] = torchnet.meter.AverageValueMeter()
        self.state['ave_emb_loss'] = torchnet.meter.AverageValueMeter()
        self.state['ap'] = torchnet.meter.AverageValueMeter()
        self.state['hammingloss'] = torchnet.meter.AverageValueMeter()
        
    def Learn(self, model, criterion, X_train, y_train, X_val=None, y_val=None, quiet_mode=False):
        if X_val is None or y_val is None:
            self.configs['evaluate'] = False
            
        if self.configs['use_multi_gpu']:
            model = torch.nn.DataParallel(model)
        model.to(self.configs['device'])
        
        # Adjacency matrix with self-loop
        # self.state['adj'] = model.label_adj.data + torch.eye(model.label_adj.data.size(0),
        #                                                      dtype=model.label_adj.data.dtype,
        #                                                      device=model.label_adj.data.device)
        self.state['adj'] = model.label_adj.data
            
        # Defining optimizer
        optimizer = torch.optim.Adam(model.get_config_optim(), lr=self.configs['lr'], weight_decay=self.configs['weight_decay'])
        
        if self.configs['evaluate']:
            self.init_eval()
        
        self.quiet_mode = quiet_mode
        if not quiet_mode:
            print('Training begins ......')
            
        for epoch in range(self.configs['start_epoch'], self.configs['max_epoch']):
            self.state['epoch'] = epoch
            self.state['lr'] = self.configs['lr']
            
            # Training for one epoch
            self.train(model, criterion, X_train, y_train, optimizer)
            
            # Evaluate on validation set
            val_metric = None
            if self.configs['evaluate']:
                with torch.no_grad():
                    val_metric = self.validate(model, criterion, X_val, y_val)
            
            # Saving checkpoint
            self.on_save_checkpoint(model, val_metric)
        
        if not quiet_mode:
            print('Training finishes.')
        
    def Step(self, training, model, criterion, X, y, optimizer=None):
        '''
        Forwarding model and optimizing model once during training.
        '''
        output, emb = model(X)
        cls_loss = criterion[0](output, y)
        if training:
            embedding_loss = criterion[1](emb, self.state['adj'])
            loss = cls_loss + self.configs['lambda'] * embedding_loss
        else:
            embedding_loss = torch.Tensor([0.0])
            loss = cls_loss 
        
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            pred_probs = output.sigmoid_()
            pred_labels = pred_probs.new_zeros(pred_probs.size())
            pred_labels[pred_probs > 0.5] = 1
            ap = AveragePrecision(pred_probs, y)
            self.state['ap'].add(ap)
            hammingloss = HammingLoss(pred_labels, y)
            self.state['hammingloss'].add(hammingloss)
            
        self.state['cur_loss'] = loss.data.item()
        self.state['ave_loss'].add(self.state['cur_loss'])
        self.state['cur_cls_loss'] = cls_loss.data.item()
        self.state['cur_emb_loss'] = embedding_loss.data.item()
        self.state['ave_cls_loss'].add(self.state['cur_cls_loss'])
        self.state['ave_emb_loss'].add(self.state['cur_emb_loss'])
    
    def on_start_epoch(self, training):
        MultiLabelEngine.on_start_epoch(self, training)
        self.state['ave_cls_loss'].reset()
        self.state['ave_emb_loss'].reset()
    
    def on_end_epoch(self, training):
        '''
        Displaying information after one epoch finishing.
        '''
        if self.configs['display'] and not self.quiet_mode:
            if training:
                print('Epoch: [{}]\t lr {:.2e}\t Average Loss {:.4f}\n'
                      'Average Cls_loss {:.4f}\t Average Emb_loss {:.4f}'.
                      format(self.state['epoch'],
                             self.state['lr'],
                             self.state['ave_loss'].value()[0],
                             self.state['ave_cls_loss'].value()[0],
                             self.state['ave_emb_loss'].value()[0]))
                with open('record_data.txt', 'a') as lf:
                    write_str = '{:.4f} {:.4f} {:.6f}\n'.format(self.state['ave_loss'].value()[0], self.state['ave_cls_loss'].value()[0],  self.state['ave_emb_loss'].value()[0] * 10000)
                    lf.write(write_str) 
            else:
                print('Test: \t Average Loss {:.4f}\t AP {:.4f}\t HammingLoss {:.4f}'.
                      format(self.state['ave_loss'].value()[0],
                             self.state['ap'].value()[0],
                             self.state['hammingloss'].value()[0]))
        
        return {'Loss': self.state['ave_loss'].value()[0],
                'AveragePrecision': self.state['ap'].value()[0],
                'HammingLoss': self.state['hammingloss'].value()[0]}
    