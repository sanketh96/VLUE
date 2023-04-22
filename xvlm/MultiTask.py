import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset, Subset

from models.model_nlvr import XVLM

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

def main(args, config):
    print("Creating dataset")
    nlvr_train_dataset, nlvr_val_dataset, nlvr_test_dataset, WIT_train_dataset, WIT_test_dataset = create_dataset('multitask', config, args.evaluate)
    nlvr_datasets = [nlvr_train_dataset, nlvr_val_dataset, nlvr_test_dataset]
    WIT_datasets = [WIT_train_dataset, WIT_test_dataset]

    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    nlvr_train_dataset_size = len(nlvr_train_dataset)
    WIT_train_dataset_size = len(WIT_train_dataset)

    
    truncated_data_size = min(nlvr_train_dataset_size, WIT_train_dataset_size)
    if truncated_data_size == nlvr_train_dataset_size:
        wit_train_dataset = Subset(wit_train_dataset, range(truncated_data_size))
    else:
        nlvr_train_dataset = Subset(nlvr_train_dataset, range(truncated_data_size))
    
    nlvr_train_dataset_size = len(nlvr_train_dataset)
    WIT_train_dataset_size = len(WIT_train_dataset)

    print("NLVR size: ", nlvr_train_dataset_size)
    print("WIT size: ", WIT_train_dataset_size)
    assert nlvr_train_dataset_size == WIT_train_dataset_size

    samplers = [None, None, None]

    nlvr_train_loader, nlvr_val_loader, nlvr_test_loader = create_loader(nlvr_datasets, samplers, batch_size=[config['batch_size_train']] * 3,
                                                              num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                              collate_fns=[None, None, None])
    WIT_train_loader, WIT_eval_loader = create_loader(WIT_datasets, samplers, batch_size=[config['batch_size_train']] * 3,
                                                              num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                              collate_fns=[None, None, None])

    for nlvr_batch_data, wit_batch_data in zip(nlvr_train_loader, WIT_train_loader):
        (image0, image1, text_nlvr, targets) = nlvr_batch_data
        (image, text_wit, idx) = wit_batch_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/MultiTask.yaml')
    parser.add_argument('--output_dir', default='output/multitask')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--load_nlvr_pretrain', action='store_true')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)