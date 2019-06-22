from os import makedirs
from os.path import isdir, join, dirname, abspath

import torch.optim as optim
from ignite.metrics import Accuracy, Precision, Recall, TopKCategoricalAccuracy, \
    MeanSquaredError, \
    MeanAbsoluteError

from hyperopt import hp
from numpy import log

from descriptors import DataSetDescriptor, ModelParameters
from networks import *

"""-------------------------------------------------------------------------"""
"""----------------------------Folder generator-----------------------------"""
"""-------------------------------------------------------------------------"""

def folder_generator(base_dir_name, dir_list=None, subdir_list=None) :
    base_dir = join(dirname(dirname(abspath(__file__))), base_dir_name)

    for folder in dir_list :
        tmp_dir = join(base_dir, folder)
        if not isdir(tmp_dir) :
            makedirs(tmp_dir)

        for subdir in subdir_list :
            tmp_subdir = join(tmp_dir, subdir)
            if not isdir(tmp_subdir) :
                makedirs(tmp_subdir)

"""-------------------------------------------------------------------------"""
"""----------------------------Model Selector-------------------------------"""
"""-------------------------------------------------------------------------"""

def model_selector(network) :
    network = network.lower()
    if network == 'mnistnet' :
        model = MNISTNet()
        parameters = model.parameters()
    elif network == 'seqmnistnet' :
        model = SequentialMNIST()
        parameters = model.parameters()
    elif network == 'cifar10net' :
        model = CIFAR10Net()
        parameters = model.parameters()
    elif network == 'lstmnet' :
        model = LSTMNet()
        parameters = model.parameters()
    elif network == 'jsbchoralesnet' :
        model = JSBChoralesNet()
        parameters = model.parameters()
    else :
        raise RuntimeError(network, " is not a valid argument!")

    return model, parameters

"""-------------------------------------------------------------------------"""
"""-----------------------Loss function selector----------------------------"""
"""-------------------------------------------------------------------------"""

def loss_fn_selector(dataset) :
    dataset = dataset.lower()
    if dataset == 'mnist' :
        loss_fn = F.nll_loss
    elif 'cifar' in dataset :
        loss_fn = F.cross_entropy
    elif 'sin' in dataset :
        loss_fn = F.mse_loss
    elif dataset == 'jsb-chorales' :
        loss_fn = F.binary_cross_entropy
    else :
        raise RuntimeError(__file__, ": ", dataset, " is not a valid argument!")

    return loss_fn

"""-------------------------------------------------------------------------"""
"""-----------------------Search space generation---------------------------"""
"""-------------------------------------------------------------------------"""

def generate_search_space(model) :
    layer_modifier = ['none']  # used for both LSTM and GRU

    model = model.lower()
    if model == 'mnistnet' :
        target_layers = ['fc', 'fc1', 'fc2', 'conv', 'conv1', 'conv2']
    elif model == 'seqmnistnet' :
        target_layers = ['lstm1', 'lstm2', 'lstm', 'fc']
        layer_modifier = ['input', 'forget', 'cell', 'output']
    elif model == 'cifar10net' :
        target_layers = ['fc', 'fc1', 'fc2', 'fc3', 'conv', 'conv1', 'conv2']
    elif model == 'lstmnet' :
        target_layers = ['lstm1', 'lstm2', 'lstm']
        layer_modifier = ['input', 'forget', 'cell', 'output']
    elif model == 'jsbchoralesnet' :
        target_layers = ['lstm1']
        layer_modifier = ['input', 'forget', 'cell', 'output']
    else :
        raise NotImplementedError(model, " is not a valid name!")

    wmm_mode = ['rect']

    space = \
        {
            'opt_target_layer' : hp.choice('target_layer', target_layers),
            'opt_layer_modifier' : hp.choice('layer_modifier', layer_modifier),
            'opt_coverage' : hp.uniform('coverage', 0.03, 0.35),
            'opt_p' : hp.loguniform('p', log(0.05), log(0.4))
            }

    return space

"""-------------------------------------------------------------------------"""
"""---------------------Generation of basic arguments-----------------------"""
"""-------------------------------------------------------------------------"""

def generate_basic_arguments(model_name, num_seeds, dataset, device, visualize) :
    model, params2optimize = model_selector(model_name)

    model_params = ModelParameters()
    optimizer = optim.SGD(params2optimize, lr=model_params.lr, momentum=model_params.momentum,
                          weight_decay=model_params.l2, nesterov=True)


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=1e-6)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 120])
    default_kwargs = \
        {
            'data_desc' : DataSetDescriptor(device=device, loss_fn=loss_fn_selector(dataset), dataset=dataset,
                                            model=model, num_seeds=num_seeds, visualize=visualize),
            'model' : model,
            'model_params' : model_params,
            'optimizer' : optimizer,
            'scheduler' : scheduler,
            'seed_idx' : 0
            }

    return default_kwargs


"""-------------------------------------------------------------------------"""
"""------------------------Test metrics selector ---------------------------"""
"""-------------------------------------------------------------------------"""

def metrics_selector(mode, loss) :
    mode = mode.lower()
    if mode == "classification" :
        metrics = {
            "loss" : loss,
            "accuracy" : Accuracy(),
            "accuracy_topk" : TopKCategoricalAccuracy(),
            "precision" : Precision(average=True),
            "recall" : Recall(average=True)
            }
    elif mode == "multiclass-multilabel" :
        metrics = {
            "loss" : loss,
            "accuracy" : Accuracy(),
            }
    elif mode == "regression" :
        metrics = {
            "loss" : loss,
            "mse" : MeanSquaredError(),
            "mae" : MeanAbsoluteError()
            }
    else :
        raise RuntimeError("Invalid task mode, select classification or regression")

    return metrics