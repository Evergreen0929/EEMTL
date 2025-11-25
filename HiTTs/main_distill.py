
import argparse
import cv2
import os
import numpy as np
import sys, json
import torch

from utils.utils import mkdir_if_missing
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_test_dataset, get_train_dataloader, get_test_dataloader,\
                                get_optimizer, get_model, get_criterion
from evaluation.evaluate_utils import PerformanceMeter
from utils.logger import Logger
from utils.train_utils import distill_train_phase, train_phase
from utils.test_utils import distill_test_phase
from termcolor import colored

from torch.utils.tensorboard import SummaryWriter
import time
start_time = time.time()

# DDP
import torch.distributed as dist
import datetime
dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(0, 3600*2))

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_exp',
                    help='Config file for the experiment', default='./configs/cityscapes/hrnet18/multi_task_baseline.yml')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
# args.local_rank = int(os.environ['LOCAL_RANK'])

print('local rank: %s' %args.local_rank)
torch.cuda.set_device(args.local_rank)

# CUDNN
torch.backends.cudnn.benchmark = True
import pdb

def set_seed(seed):
    # Stop randomness in each process in the DDP
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(params):
    set_seed(0)
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_exp, params)

    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
    if args.local_rank == 0:
        print(colored(p, 'red'))

    # Init performance meter
    _ = PerformanceMeter(p, [t for t in p.TASKS.NAMES if t != '3ddet'])

    models = []
    criterions = []
    schedulers = []
    optimizers = []
    for gi in range(p['global_iter']):
        # Get model
        if args.local_rank == 0:
            print(colored('Initializing Model {}'.format(gi + 1), 'blue'))
        model = get_model(p)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=False)

        # Get criterion
        criterion = get_criterion(p)

        # Optimizer
        scheduler, optimizer = get_optimizer(p, model)

        models.append(model)
        criterions.append(criterion)
        schedulers.append(scheduler)
        optimizers.append(optimizer)

    # Transforms
    train_transforms, val_transforms = get_transformations(p)
    if params['run_mode'] != 'infer':
        train_dataset = get_train_dataset(p, train_transforms)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_dataloader = get_train_dataloader(p, train_dataset, train_sampler)
    test_dataset = get_test_dataset(p, val_transforms)
    test_dataloader = get_test_dataloader(p, test_dataset)

    # Resume from checkpoint
    if os.path.exists(p['checkpoint']) or params['run_mode'] == 'infer':
        if args.local_rank == 0:
            print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        start_global_iter = checkpoint['global_iter']
        optimizers[start_global_iter].load_state_dict(checkpoint['optimizer'])
        schedulers[start_global_iter].load_state_dict(checkpoint['scheduler'])
        models[checkpoint['global_iter']].load_state_dict(checkpoint['model'])
        if start_global_iter > 0:
            teacher_ckpt = checkpoint
            models[checkpoint['global_iter'] - 1].load_state_dict(teacher_ckpt['model'])

        start_epoch = checkpoint['epoch'] + 1
        iter_count = checkpoint['iter_count']  # already + 1 when saving
    else:
        if args.local_rank == 0:
            print(colored('Fresh start...', 'blue'))
        start_epoch = 0
        iter_count = 0
        start_global_iter = 0

    if DEBUG_FLAG and args.local_rank == 0:
        print("\nFirst Testing...")
        eval_test = distill_test_phase(p, test_dataloader, model, criterion, iter_count)
        print(eval_test)
    
    # Main loop
    if params['run_mode'] != 'infer':
        for gi in range(start_global_iter, p['global_iter']):

            # Initialization TB Writing
            tb_log_dir = './RESULTS/' + p.version_name + '/tb_dir' + '_stage{}'.format(gi + 1)  # os.path.join(p['output_dir'], 'tensorboard_logdir')
            p.tb_log_dir = tb_log_dir
            if args.local_rank == 0:
                train_tb_log_dir = tb_log_dir + '/train'
                test_tb_log_dir = tb_log_dir + '/test'
                if params['run_mode'] != 'infer':
                    mkdir_if_missing(tb_log_dir)
                    mkdir_if_missing(train_tb_log_dir)
                    mkdir_if_missing(test_tb_log_dir)
                tb_writer_train = SummaryWriter(train_tb_log_dir)
                tb_writer_test = SummaryWriter(test_tb_log_dir)
                print(f"Tensorboard dir: {tb_log_dir}")
            else:
                tb_writer_train = None
                tb_writer_test = None

            # Model Weight Inherent
            if params['model_weight_inherent']:
                if gi > 0:
                    print("Copy model weights from Model {} to Model {}".format(gi - 1, gi))
                    models[gi].load_state_dict(models[gi - 1].state_dict(), strict=True)

            for epoch in range(start_epoch, p['epochs']):
                train_sampler.set_epoch(epoch)
                if args.local_rank == 0:
                    print(colored('Global Stage %d/%d' % (gi + 1, p['global_iter']), 'yellow'))
                    print(colored('Epoch %d/%d' % (epoch + 1, p['epochs']), 'yellow'))
                    print(colored('-' * 10, 'yellow'))

                if gi > 0:
                    end_signal, iter_count = distill_train_phase(p, args, train_dataloader, test_dataloader, None,
                                                                 models[gi],
                                                                 criterions[gi],
                                                                 optimizers[gi], schedulers[gi], epoch, gi,
                                                                 tb_writer_train,
                                                                 tb_writer_test,
                                                                 iter_count, teacher_model=models[gi - 1])
                else:
                    end_signal, iter_count = distill_train_phase(p, args, train_dataloader, test_dataloader, None,
                                                                 models[gi], criterions[gi],
                                                                 optimizers[gi], schedulers[gi], epoch, gi,
                                                                 tb_writer_train,
                                                                 tb_writer_test,
                                                                 iter_count)
                    # end_signal, iter_count = train_phase(p, args, train_dataloader, test_dataloader, models[gi], criterions[gi],
                    #                                      optimizers[gi], schedulers[gi], epoch, tb_writer_train, tb_writer_test,
                    #                                      iter_count)
                schedulers[gi].step()

                if end_signal:
                    if gi != p['global_iter'] - 1:
                        iter_count = 0
                    break

    # Evaluate best model at the end
    # running eval
    if args.local_rank == 0:
        if p.run_mode == 'infer' or True:
            # print('Infer at batch {}'.format(start_epoch))
            if p.run_mode == 'train':
                print('Checkpoint ...')
                torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                            'epoch': epoch, 'iter_count': iter_count-1, 'global_iter': gi}, p['checkpoint'])
            print('Infer at iteration {}'.format(iter_count))
            eval_epoch = iter_count  # start_epoch
            eval_test = distill_test_phase(p, test_dataloader, models[-1], criterions[-1], eval_epoch, global_iter=p['global_iter'] - 1, save_result=True)
            print('Infer test restuls:')
            print(eval_test)

        end_time = time.time()
        run_time = (end_time-start_time) / 3600
        print('Total running time: {} h.'.format(run_time))

if __name__ == "__main__":
    params = {}
    params['version_name'] = 'HiTTs_MS_Distill_one_nyud_final'
    # params['version_name'] = 'HiTTs_MS_Distill_one_pascal_final'

    # IMPORTANT VARIABLES
    params["semseg_save_train_class"] = False
    params['run_mode'] = 'train'
    params['model_weight_inherent'] = False

    DEBUG_FLAG = False 

    args.config_exp = './configs/nyud/hitts.yml'
    # args.config_exp = './configs/pascal/hitts.yml'
    main(params)
