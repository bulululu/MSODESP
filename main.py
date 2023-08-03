import os
import sys
import random
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data.dataset import random_split
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from utils.tools import get_logger
from lib.clinicalscore import PreDataset
from lib.dataset import SepsisDataset
from lib.model import *

'''
'An Explanatory Multiscale Neural Differential Equation That Can '
                                 'Handle Missing Data for Multi center Sepsis Prediction'
'''

parser = argparse.ArgumentParser('Sepsis Prediction')

# parameters for data center.
parser.add_argument('-path', '--read_path', type=str, default='./data/source/sepsis.csv',
                    help='The path of dataset to load')
parser.add_argument('-r', '--random-seed', type=int, default=2, help="random seed")
parser.add_argument('--internal_center', default=['mimic3cv', 'mimiciv', 'eicu'], type=str, nargs='+',
                    help="Different medical center to load, available: eicu, mimic3cv, mimiciv, xjtu")
parser.add_argument('--external_center', default=['xjtu'], type=str, nargs='+',
                    help="Different medical center to load, available: eicu, mimic3cv, mimiciv, xjtu")
parser.add_argument('--is_external_validation', action='store_false', help='whether to use the external validation')

# parameters for pre-processing medical time series.
parser.add_argument('--merge_time_window', type=int, default=8,
                    help='The unit of the time window to merge the features, unit: hour')
parser.add_argument('--sample_time_window', type=int, default=2,
                    help='The unit of the feature sampling, unit: hour')
parser.add_argument('--predict_time_window', type=int, default=8,
                    help='The length of the time window to prediction, unit: hour')
parser.add_argument('--adopt_time_window', type=int, default=24,
                    help='The number of the time window for training data, related to the merge_time_window')
parser.add_argument('--method_merge', type=str, nargs='+', default=['last', 'min', 'max'])
parser.add_argument('--threshold_missing', type=float, default=0.5, help='Determine multi-scale time threshold')

# parameters for training the model.
parser.add_argument('--cuda', action='store_false', help='whether to use cuda')
parser.add_argument('--epoch', type=int, default=100, help='the epoch of training the model')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate to update the parameters')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--train_size', type=float, default=0.7, help='train size')
parser.add_argument('--valid_size', type=float, default=0.1, help='valid size')
parser.add_argument('--test_size', type=float, default=0.2, help='test size')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim')

args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')
    experiment_id = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_path = f'./logs/run_models_{experiment_id}.log'
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(' '.join(sys.argv))
    center_internal = '_'.join(args.internal_center)
    path_internal = f'./data/processed/sepsis_merged_{args.merge_time_window}_{args.sample_time_window}_' \
                    f'{args.threshold_missing}_{center_internal}.pt'
    center_external = '_'.join(args.external_center)
    path_external = f'./data/processed/sepsis_merged_{args.merge_time_window}_{args.sample_time_window}_' \
                    f'{args.threshold_missing}_{center_external}.pt'
    logger.info('internal dataset: ' + path_internal)
    logger.info('')
    if args.cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if not os.path.exists(path_internal) or not os.path.exists(path_external):
        pre_sepsis = PreDataset(args)

    data_min, data_max = torch.load('./data/data_min_max.pt', map_location=device)
    dataset_internal = SepsisDataset(args,
                                     path_internal,
                                     device=device,
                                     normalization=True,
                                     data_min=data_min,
                                     data_max=data_max)
    logger.info(dataset_internal)

    sepsis_model = SepsisLSTM(in_dim=215, hidden_dim=args.hidden_dim, n_layer=2, n_classes=1).to(device)
    sepsis_optim = optimizer.Adam(lr=args.learning_rate, params=sepsis_model.parameters())
    loss = compute_loss()
    with SummaryWriter(log_dir=f'./tensorboard/{experiment_id}') as writer:
        model, auc_train, auc_valid = train(args=args,
                                            model=sepsis_model,
                                            optimizer=sepsis_optim,
                                            loss=loss,
                                            trainloader=dataset_internal.trainloader,
                                            validloader=dataset_internal.validloader,
                                            epoch=args.epoch,
                                            logger=logger,
                                            ckpt_path=f'./experiments/{experiment_id}/',
                                            writer=writer)

        model, auc_test = test(model=model,
                               loss=loss,
                               testloader=dataset_internal.testloder,
                               logger=logger,
                               writer=writer,
                               external_center='internal test',
                               ckpt_path=f'./experiments/{experiment_id}/',
                               state='test')

        if args.is_external_validation:
            logger.info('external dataset: ' + path_external)
            external_data = SepsisDataset(args,
                                          path_external,
                                          is_external=True,
                                          normalization=True,
                                          data_min=dataset_internal.data_min,
                                          data_max=dataset_internal.data_max,
                                          device=device)
            logger.info(external_data)
            model, auc_external = test(model=model,
                                       loss=loss,
                                       testloader=external_data.external_loader,
                                       logger=logger,
                                       writer=writer,
                                       external_center='external xjtu',
                                       ckpt_path=f'./experiments/{experiment_id}/',
                                       state='external')

        logger.info('-------------------Summary---------------------')
        logger.info(f'train best auc: {auc_train:.6f}')
        logger.info(f'valid best auc: {auc_valid:.6f}')
        logger.info(f' test best auc: { auc_test:.6f}')
        if args.is_external_validation:
            logger.info(f'exter best auc: {auc_external:6f}')
