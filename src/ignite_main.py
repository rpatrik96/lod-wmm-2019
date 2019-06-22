"""
Created on Sun Oct 15 13:02:05 2017

@author: Patrik Reizinger

@brief:
   This piece of code implements the framework to experiment with 2 new regularization techniques, Weight Shuffling and
   Weight Reinitialization - called Weight Matrix Modification (WMM)
"""
from __future__ import print_function

import argparse
from os import makedirs, listdir
from os.path import isdir

import gc
import torch
from hyperopt import fmin, tpe
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss

from ignite_callbacks import *
from utilities import folder_generator, generate_basic_arguments, generate_search_space, metrics_selector
from wmm import WeightShuffle, WeightReinit

def objective_fn(kwargs):
    """Objective function for hyperopt"""

    """------------------------"""
    """Argument parsing"""
    # hyperopt only passes 1 argument to the objective function
    opt_p = kwargs['opt_p']
    opt_target_layer = kwargs['opt_target_layer']
    opt_layer_modifier = kwargs['opt_layer_modifier']
    opt_coverage = kwargs['opt_coverage']
    model = kwargs['model']
    model_params = kwargs['model_params']
    optimizer = kwargs['optimizer']
    scheduler = kwargs['scheduler']
    data_desc = kwargs['data_desc']
    seed_idx = kwargs['seed_idx']
    choose_reinit = kwargs['choose_reinit']

    # names setup
    wmm_namemodifier = 'reinit' if choose_reinit else 'shuffle'

    model_dir = join(join(join(data_desc.base_dir, 'models'), wmm_namemodifier), data_desc.timestamp)
    time_series_dir = join(join(join(data_desc.base_dir, 'time_series'), wmm_namemodifier), data_desc.timestamp)
    results_dir = join(join(data_desc.base_dir, 'results'), wmm_namemodifier)

    for d in [model_dir, time_series_dir]:
        if not isdir(d):
            makedirs(d)

    num_model_dir_items = [isfile(file) for file in listdir(model_dir)].count(True)
    model_idx = num_model_dir_items / 2 if args.checkpoint else num_model_dir_items  # divide by 2 because of checkpointing

    model_path = join(model_dir, 'model_' + str(len(listdir(time_series_dir))) + 's' + str(seed_idx))
    time_series_path = join(time_series_dir, 'time_series_' + str(len(listdir(time_series_dir))))
    df_path = join(results_dir, wmm_namemodifier + '_' + data_desc.timestamp + '.tsv')

    plot_var_name = str(model_idx) + "_s" + str(seed_idx)  # for plotter

    # generate DataLoader objects
    train_loader, valid_loader, test_loader = data_desc.generate_datasets(seed_idx, model_params.batch_size,
                                                                          model_params.test_batch_size,
                                                                          data_desc.device)

    """------------------------"""
    """WMM"""
    WmmObject = WeightReinit if choose_reinit else WeightShuffle
    WmmObject = WmmObject(p=opt_p, coverage=opt_coverage, layer=opt_target_layer, layer_modifier=opt_layer_modifier)

    """------------------------"""
    """Engines"""
    loss = Loss(data_desc.loss_fn)
    # trainer_adam = create_supervised_trainer(model, adam, data_desc.loss_fn, data_desc.device)
    trainer = create_supervised_trainer(model, optimizer, data_desc.loss_fn, data_desc.device)
    evaluator = create_supervised_evaluator(model, metrics={"loss": loss}, device=data_desc.device)
    tester = create_supervised_evaluator(model, metrics_selector(data_desc.problem_type, loss), device=data_desc.device)

    """------------------------"""
    """Handlers"""

    trainer.add_event_handler(Events.STARTED, IgniteCustomLogger().log_init, args.interval)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, wmm_callback, model, WmmObject)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, IgniteCustomLogger().log_validate, evaluator, valid_loader,
                              plot_var_name)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, IgniteCustomLogger().lr_scheduling, scheduler)
    trainer.add_event_handler(Events.COMPLETED, IgniteCustomLogger().log_test, tester, test_loader, model, model_params,
                              WmmObject, df_path, time_series_path)

    """Early stopping"""
    earlyStoppingHandler = EarlyStopping(patience=model_params.patience, score_function=early_stopping_score,
                                         trainer=trainer)

    evaluator.add_event_handler(Events.COMPLETED, earlyStoppingHandler)

    """Checkpointing"""
    if args.checkpoint:
        # must be defined here, because the handler should be added to the trainer, and the function takes only one argument
        # which is the trainer engine

        def checkpoint_score(engine):
            return -evaluator.state.metrics["loss"]  # improvement = higher score

        checkPointHandler = ModelCheckpoint(dirname=model_dir, filename_prefix=data_desc.dataset,
                                            score_function=checkpoint_score, score_name="val_loss", require_empty=False,
                                            save_as_state_dict=True, n_saved=1)

        trainer.add_event_handler(Events.ITERATION_COMPLETED, checkPointHandler,
                                  {str(model_idx) + "s" + str(seed_idx): model})

    """------------------------"""
    """Run"""
    trainer.run(train_loader, max_epochs=model_params.num_epochs)

    return tester.state.metrics["loss"]

if __name__ == "__main__":

    """-------------------------------------------------------------------------"""
    """--------------------------Command line arguments-------------------------"""
    """-------------------------------------------------------------------------"""
    parser = argparse.ArgumentParser(description='Weight Matrix Modification hyperoptimization')

    parser.add_argument('--gen-rand', action='store_true', default=False,
                        help='generates random indices')
    parser.add_argument('--checkpoint', action='store_true', default=False,
                        help='Save the best model to disk')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='whether to display training process state data')
    parser.add_argument('--choose-reinit', action="store_true", default=False,
                        help='switch between reinit and shuffle (default: shuffle)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-seeds', type=int, default=1, metavar='NS',
                        help='number of seeds (max and default: 1)')
    parser.add_argument('--num-trials', type=int, default=1, metavar='T',
                        help='number of steps for the TPE (default: 1)')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                        help='dataset (default: mnist)')
    parser.add_argument('--model', type=str, default='mnistnet', metavar='M',
                        help='model (default: mnistnet)')
    parser.add_argument('--interval', type=int, default=10, metavar='I',
                        help='interval for logging&validation (default: 10)')

    # Argument parsing
    args = parser.parse_args()

    if args.visualize and not visdom.Visdom().check_connection():
        raise RuntimeError("Visdom not running, start a server!")

    gc.enable()  # Garbage collection

    """Random seed"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    """"Device & dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """-------------------------------------------------------------------------"""
    """------------------------------File structure-----------------------------"""
    """-------------------------------------------------------------------------"""
    folder_generator(join('datasets', args.dataset), dir_list=['results', 'time_series', 'models'],
                     subdir_list=['shuffle', 'reinit'])

    space = {
        **generate_search_space(args.model),
        **generate_basic_arguments(args.model, 1, args.dataset, device, args.visualize)
        }

    """-------------------------------------------------------------------------"""
    """----------------------------------Run------------------------------------"""
    """-------------------------------------------------------------------------"""
    fmin(fn=objective_fn, space={**space, 'choose_reinit': args.choose_reinit}, algo=tpe.suggest,
         max_evals=args.num_trials, return_argmin=False)
