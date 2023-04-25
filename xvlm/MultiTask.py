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
from models.model_multitask import MultiTask

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer



def train(model, nlvr_data_loader, wit_data_loader, optimizer, tokenizer, epoch, device, scheduler):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100

    nlvr_iterator = metric_logger.log_every(nlvr_data_loader, print_freq, header)
    wit_iterator = metric_logger.log_every(wit_data_loader, print_freq, header)

    for nlvr_elements, wit_elements in zip(nlvr_iterator, wit_iterator):

        wit_image, wit_text, wit_targets = wit_elements

        images = wit_image.to(device)
        targets = wit_targets.to(device)   
        text_inputs = tokenizer(wit_text, padding='longest', return_tensors="pt").to(device)  
        loss_itc, loss_itm = model(image=images, text_ids=text_inputs.input_ids,
                                   text_atts=text_inputs.attention_mask, targets=targets,
                                   task='contrastive', train=True)
        
        ####### NLVR #######
        nlvr_image0, nlvr_image1, nlvr_text, nlvr_targets = nlvr_elements
        images = torch.cat([nlvr_image0, nlvr_image1], dim=0)
        images = images.to(device)
        targets = nlvr_targets.to(device)   
        text_inputs = tokenizer(nlvr_text, padding='longest', return_tensors="pt").to(device)  

        loss_ce = model(image=images, text_ids=text_inputs.input_ids,
                                   text_atts=text_inputs.attention_mask, targets=targets,
                                   task='classification', train=True)

        loss = loss_itc + loss_itm + loss_ce
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):

    # Initialize distributed training
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size'] = args.bs // world_size

    # Set seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Create dataset
    nlvr_train_dataset, nlvr_val_dataset, nlvr_test_dataset, WIT_train_dataset, WIT_test_dataset = \
        create_dataset('multitask', config, args.evaluate)
    nlvr_datasets = [nlvr_train_dataset, nlvr_val_dataset, nlvr_test_dataset]
    WIT_datasets = [WIT_train_dataset, WIT_test_dataset]

    # Create model
    model = MultiTask(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Create model with\without DDP (Distributed Data Parallel)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model
        
    # Create tokenizer
    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([test_dataset], [False], num_tasks, global_rank)
        else:
            samplers = [None] * 3

        # TODO: Workers = 4
        nlvr_test_loader = create_loader([nlvr_test_dataset], samplers=samplers,
                                         batch_size=[config['batch_size_train']] * 3,
                                         num_workers=[1, 1, 1], is_trains=[True, False, False],
                                         collate_fns=[None] * 3)[0]
        WIT_eval_loader = create_loader([WIT_test_dataset], samplers=samplers,
                                        batch_size=[config['batch_size_train']] * 3,
                                        num_workers=[1, 1, 1], is_trains=[True, False, False],
                                        collate_fns=[None] * 3)[0]

        test_stats = evaluate(model, nlvr_test_loader, WIT_eval_loader, tokenizer, device)

        if utils.is_main_process():
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
            print(log_stats)

        dist.barrier()

    else:
        print("Start training")

        train_dataset_size = len(nlvr_train_dataset)
        train_batch_size = config['batch_size_train']
        world_size = utils.get_world_size()

        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")
            print(f"### test data {len(nlvr_test_dataset)}", flush=True)
            print(f"### test data {len(WIT_test_dataset)}", flush=True)

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
        else:
            samplers = [None] * 3

        nlvr_train_loader, nlvr_val_loader, nlvr_test_loader = create_loader(nlvr_datasets, samplers=samplers,
                                                              batch_size=[config['batch_size_train']] * 3,
                                                              num_workers=[1, 1, 1], is_trains=[True, False, False],
                                                              collate_fns=[None] * 3)
        wit_train_loader, wit_test_loader = create_loader(WIT_datasets, samplers=[None] * 2,
                                                              batch_size=[config['batch_size_train']] * 2,
                                                              num_workers=[1, 1], is_trains=[True, False],
                                                              collate_fns=[None] * 2)

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(train_batch_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']

        best = 0
        best_epoch = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                nlvr_train_loader.sampler.set_epoch(epoch)
                wit_train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, nlvr_train_loader, wit_train_loader, optimizer, tokenizer, epoch, device, lr_scheduler)
            val_stats = evaluate(model, val_loader, tokenizer, device)
            test_stats = evaluate(model, test_loader, tokenizer, device)

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                if float(val_stats['acc']) > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = float(val_stats['acc'])
                    best_epoch = epoch

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))

# def main2(args, config):    
#     print("Creating dataset")
#     nlvr_train_dataset, nlvr_val_dataset, nlvr_test_dataset, WIT_train_dataset, WIT_test_dataset = create_dataset('multitask', config, args.evaluate)
#     nlvr_datasets = [nlvr_train_dataset, nlvr_val_dataset, nlvr_test_dataset]
#     WIT_datasets = [WIT_train_dataset, WIT_test_dataset]

#     if config['use_roberta']:
#         tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
#     else:
#         tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

#     samplers = [None, None, None]

#     # TODO: Workers = 4

#     nlvr_train_loader, nlvr_val_loader, nlvr_test_loader = create_loader(nlvr_datasets, samplers, batch_size=[config['batch_size_train']] * 3,
#                                                               num_workers=[1, 1, 1], is_trains=[True, False, False],
#                                                               collate_fns=[None, None, None])
#     WIT_train_loader, WIT_eval_loader = create_loader(WIT_datasets, samplers, batch_size=[config['batch_size_train']] * 3,
#                                                               num_workers=[1, 1, 1], is_trains=[True, False, False],
#                                                               collate_fns=[None, None, None])
    
#     print("Creating model")
#     model = MultiTask(config=config)
#     model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
#     model = model.to(device)
#     print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))




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