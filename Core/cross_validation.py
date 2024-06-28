import torchnet
import os
import torch

def cross_validation(model, dataset, data, nfold=1, shuffle=True, random_state=None,
                     eval_on_trainset=False, quiet_mode=True, ada_epoch=True,
                     save_model=True):
    '''
    Evaluating model by cross-validation

    Parameters
    ----------
    model :
        An instance of a learning model waiting for cross-validation, which
        should has 'Train', 'Test' and 'reset_parameters' functions realized.
    dataset : 
        An instance of dataset on which cross-validation is conducted.
    data : 
        Data List.
    nfold : int, optional
        Number of fold for cross-validation. The default is 10.
    shuffle : bool, optional
        If shuffle=True, shuffling the data before cross-validation.
        The default is True.
    random_state : int, optional
        When shuffle is True, random_state affects the ordering of the indices,
        which controls the randomness of each fold. Otherwise, this parameter
        has no effect. The default is None.
    eval_on_trainset: bool, optional
        If eval_on_trainset=True, also evaluate on the training set.
        The default is False.
    quiet_mode : bool, optional
        If quiet_mode=True, training information will not be displayed during
        parameter searching.
        The default is True.
    ada_epoch : bool, optional
        If ada_epoch=True, maximum training epochs will be decided on the first fold.
        The default is True.
    save_model : bool, optional
        If save_model=True, save models for each fold.
        The default is True.

    Returns
    -------
    val_metrics : dict
        Metrics on the validation set. Metrics returned are determined by model's
        'Test' function.
    train_metrics: dict
        Metrics on the training set, only valid when eval_on_trainset=True. 
        Metrics returned are determined by model's 'Test' function.
    '''
    # Creating metric recorder
    val_metrics = {}
    for metric in model.configs['label_metrics']:
        val_metrics[metric] = torchnet.meter.AverageValueMeter()
    for metric in model.configs['score_metrics']:
        val_metrics[metric] = torchnet.meter.AverageValueMeter()
    val_metrics["Precision"] = torchnet.meter.AverageValueMeter()
    val_metrics["Recall"] = torchnet.meter.AverageValueMeter()
    val_metrics["Mrr"] = torchnet.meter.AverageValueMeter()
    val_metrics["Rmse"] = torchnet.meter.AverageValueMeter()
        
    if eval_on_trainset:
        train_metrics = {}
        for metric in model.configs['label_metrics']:
            train_metrics[metric] = torchnet.meter.AverageValueMeter()
        for metric in model.configs['score_metrics']:
            train_metrics[metric] = torchnet.meter.AverageValueMeter()
    else:
        train_metrics = None
        
    # _, _, X_train, X_test, y_train, y_test, label_adj = dataset.Data()
    X_train, X_test, y_train, y_test, adj = data
    
    for count in range(1, nfold+1):
        print('Cross-validation: [{}/{}].'.format(count, nfold))
        
        # train_inds = range(X.size(0)//2)
        # test_inds = range(X.size(0)//2, X.size(0))
        # train_inds, test_inds = dataset.DataCVSplitter(count, nfold, shuffle, random_state)
        # X_train = X[train_inds]
        # y_train = y[train_inds]
        # X_test = X[test_inds]
        # y_test = y[test_inds]
        
        if ada_epoch and count == 1:
            # Training with evaluation
            model.configs['evaluate'] = True
            model.reset_parameters()
            model.configs['best_epoch'] = 0
            model.Train(X_train, y_train, adj, X_test, y_test, quiet_mode=quiet_mode)
            model.Load_checkpoint(model.configs['best_checkpoint_path'])
            model.configs['start_epoch'] = 0
            model.configs['max_epoch'] = model.configs['best_epoch']
            model.configs['evaluate'] = False
        else:
            # Training
            model.reset_parameters()
            model.Train(X_train, y_train, adj, quiet_mode=quiet_mode)
        
        # Testing
        test_metrics = model.Test(X_test, y_test)
        for key in test_metrics:
            val_metrics[key].add(test_metrics[key])
        
        if eval_on_trainset:
            test_metrics = model.Test(X_train, y_train)
            for key in test_metrics:
                train_metrics[key].add(test_metrics[key])
        
        # Saving models
        if save_model:
            if not os.path.exists(model.configs['save_checkpoint_path']):
                os.makedirs(model.configs['save_checkpoint_path'])
            fileName = 'checkpoint_{:d}_{:d}_{:d}_cv'.format(shuffle, random_state, nfold)
            path = os.path.join(model.configs['save_checkpoint_path'],
                                fileName+'{:d}.pth'.format(count))
            save_checkpoint({'epoch': model.configs['max_epoch'],
                             'state_dict': model.model.module.state_dict() 
                             if model.configs['use_multi_gpu']
                             else model.model.state_dict()}, path)

    return val_metrics, train_metrics

def save_checkpoint(checkpoint, path):
    torch.save(checkpoint, path)