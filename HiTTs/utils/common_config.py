import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil

import pdb

from easydict import EasyDict as edict
def edict2dict(inp):
    if type(inp) == dict:
        return {k: edict2dict(v) for k, v in inp.items()}
    elif type(inp) == edict:
        return {k: edict2dict(v) for k, v in inp.items()}
    else:
        return inp

"""
    Model getters 
"""
def get_backbone(p):
    """ Return the backbone """

    if p['backbone'] == 'resnet18':
        from models.resnet import resnet18
        backbone = resnet18(p['backbone_kwargs']['pretrained'])
        backbone_channels = [64, 128, 256, 512]
        p.backbone_channels = backbone_channels
        backbone_stride = [4, 8, 16, 32]
        img_h, img_w = p.TRAIN.SCALE
        p.spatial_dim = [[img_h//st, img_w//st] for st in backbone_stride]


    elif p['backbone'] == 'vitB':
        from models.transformers.vit import vit_base_patch16_384
        backbone = vit_base_patch16_384(pretrained=True, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = [768 for _ in range(4)]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0] // 16, p.TRAIN.SCALE[1] // 16] for _ in range(4)]
        # p.final_embed_dim = p.embed_dim + p.PRED_OUT_NUM_CONSTANT
    
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    """ Return the decoder head """

    if p['head'] == 'mlp':
        from models.mlphead import MLPHead
        return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    elif p['head'] == 'deeplab':
        from models.prompt_aspp import DeepLabHead
        return DeepLabHead(p, backbone_channels[-1], p.TASKS.NUM_OUTPUT[task], p.TASKS.NUM_TOKEN[task])
    else:
        raise NotImplementedError


def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)

    if p['setup'] == 'multi_task':
        if p['model'] == 'prompt_baseline':
            from models.models import MultiTaskPrompter
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MultiTaskPrompter(p, backbone, heads, p.TASKS.NAMES, feat_dim=backbone_channels)

        elif p['model'] == 'prompt_baseline_ms':
            from models.models import MTMSPrompter
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MTMSPrompter(p, backbone, heads, p.TASKS.NAMES, feat_dim=backbone_channels)

        else:
            raise NotImplementedError('Unknown model {}'.format(p['model']))
    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))
    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] == 'PASCALContext' or p['train_db_name'] == 'NYUD':
        train_transforms = torchvision.transforms.Compose([ # borrowed from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        raise ValueError('Invalid train db name'.format(p['train_db_name']))


def get_train_dataset(p, transforms=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p, p.db_paths['PASCALContext'], download=False, split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in p.ALL_TASKS.NAMES,
                                          do_edge='edge' in p.ALL_TASKS.NAMES,
                                          do_normals='normals' in p.ALL_TASKS.NAMES,
                                          do_sal='sal' in p.ALL_TASKS.NAMES,
                                          do_human_parts='human_parts' in p.ALL_TASKS.NAMES,
                                          overfit=p['overfit'])

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p, p.db_paths['NYUD_MT'], split='train', transform=transforms, retname=True,
                                    do_semseg='semseg' in p.TASKS.NAMES,
                                    do_normals='normals' in p.TASKS.NAMES,
                                    do_depth='depth' in p.TASKS.NAMES, overfit=p['overfit'])

    return database


def get_train_dataloader(p, dataset, sampler):
    """ Return the train dataloader """
    collate = collate_mil

    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True, sampler=sampler)
    return trainloader


def get_test_dataset(p, transforms=None):
    """ Return the test dataset """

    db_name = p['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p, p.db_paths['PASCALContext'], download=False, split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_edge='edge' in p.TASKS.NAMES,
                                      do_normals='normals' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                      overfit=p['overfit'])

    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p, p.db_paths['NYUD_MT'], split='val', transform=transforms,
                                    do_semseg='semseg' in p.TASKS.NAMES,
                                    do_normals='normals' in p.TASKS.NAMES,
                                    do_depth='depth' in p.TASKS.NAMES)
    
    else:
        raise NotImplemented("test_db_name")

    return database


def get_test_dataloader(p, dataset):
    """ Return the validation dataloader """
    if p['val_db_name'] == 'kitti':
        from mmcv.parallel import collate
    else:
        collate = collate_mil

    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader


""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index) 

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(ignore_index=p.ignore_index)

    else:
        criterion = None

    return criterion


def get_criterion(p):
    """ Return training criterion for a given setup """

    if p['setup'] == 'multi_task':
        if p['loss_kwargs']['loss_scheme'] == 'baseline': # Fixed weights
            from losses.loss_schemes import SemiSupervisedMultiTaskLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return SemiSupervisedMultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)
        elif p['loss_kwargs']['loss_scheme'] == 'baseline_token': # Fixed weights
            from losses.loss_schemes import SemiSupervisedMultiTaskTokenLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return SemiSupervisedMultiTaskTokenLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)
        else:
            raise NotImplementedError('Unknown loss scheme {}'.format(p['loss_kwargs']['loss_scheme']))

    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    print('Optimizer uses a single parameter group - (Default)')
    params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))


    # get scheduler
    if p.scheduler == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(optimizer, p.epochs, gamma=0.9, min_lr=0)
    elif p.scheduler == 'step':
        scheduler = torch.optim.MultiStepLR(optimizer, milestones=p.scheduler_kwargs.milestones, gamma=p.scheduler_kwargs.lr_decay_rate)

    return scheduler, optimizer

