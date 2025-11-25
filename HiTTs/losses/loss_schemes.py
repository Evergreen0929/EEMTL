import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pdb
import numpy as np
from losses.loss_functions import Spatial_BalancedBinaryCrossEntropyLoss, Spatial_L1Loss, Spatial_CrossEntropyLoss, CrossEntropyLoss
from models.normal_clusters import normal_clu_center, depth_linear_clu_center, depth_linear_clu_center_citys


class SemiSupervisedMultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(SemiSupervisedMultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

        self.mse =  torch.nn.MSELoss()

        ##### get task_mask ####
        ssl_type = p.ssl_type
        if p.train_db_name == 'PASCALContext':
            labelroot = './data/ssl_mapping/pascal/new_'
            if ssl_type == 'onelabel':
                self.labels_weights = torch.load('{}onelabel.pth'.format(labelroot))
            elif ssl_type == 'randomlabels':
                self.labels_weights = torch.load('{}randomlabels.pth'.format(labelroot))
        elif p.train_db_name == 'Cityscapes':
            labelroot = './data/ssl_mapping/cityscapes/'
            self.labels_weights = torch.load('{}onelabel.pth'.format(labelroot))['labels_weights'].float().cuda()
        elif p.train_db_name == 'NYUD':
            labelroot = './data/ssl_mapping/nyud_low/'
            if ssl_type == 'onelabel':
                self.labels_weights = torch.load('{}onelabel.pth'.format(labelroot))['labels_weights'].float().cuda()
            elif ssl_type == 'randomlabels':
                self.labels_weights = torch.load('{}randomlabels.pth'.format(labelroot))['labels_weights'].float().cuda()

        self.inter_loss_weight = 0.5
    
    def forward(self, pred, gt, tasks):
        inter_pred = pred['inter_preds']
        pred = pred['preds']
        out = {task: 0 for task in tasks}
        out.update({'inter_'+task: 0 for task in tasks})
        out['total'] = 0

        # for paritial labels available
        local_bs = gt['semseg'].shape[0]

        image_index = gt['meta']['img_name']

        for idx in range(local_bs):
            w = self.labels_weights[image_index[idx]] #.clone().float().cuda()
            label_count = 0
            sample_loss = 0
            for t_idx, task in enumerate(tasks):
                if w[t_idx] == 1:
                    label_count += 1
                    cur_loss = self.loss_ft[task](pred[idx][task], gt[task][idx:idx+1]) / local_bs 
                    out[task] += cur_loss.detach()
                    sample_loss += cur_loss * self.loss_weights[task]

                    # intermediate supervision
                    cur_loss = self.loss_ft[task](inter_pred[task][idx:idx+1], gt[task][idx:idx+1]) / local_bs 
                    out['inter_'+task] += cur_loss.detach()
                    sample_loss += cur_loss * self.loss_weights[task] * self.inter_loss_weight

            # balance loss scale
            if label_count == 0:
                pass
            else:
                out['total'] += sample_loss * len(tasks) / label_count

        return out


class SemiSupervisedMultiTaskTokenLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(SemiSupervisedMultiTaskTokenLoss, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

        if 'normals' in self.tasks:
            self.normal_set = normal_clu_center().cuda()

        if 'depth' in self.tasks:
            if len(self.tasks) == 2:
                self.depth_set = depth_linear_clu_center_citys(50).cuda()
            else:
                self.depth_set = depth_linear_clu_center(30).cuda()

        ##### get task_mask ####
        ssl_type = p.ssl_type
        if p.train_db_name == 'PASCALContext':
            labelroot = './data/ssl_mapping/pascal/'
            if ssl_type == 'onelabel':
                self.save_dict = np.load('{}onelabel.npy'.format(labelroot), allow_pickle=True).item()
            elif ssl_type == 'randomlabels':
                self.save_dict = np.load('{}randomlabels.npy'.format(labelroot), allow_pickle=True).item()
        elif p.train_db_name == 'NYUD':
            labelroot = './data/ssl_mapping/nyud_low/'
            if ssl_type == 'onelabel':
                labels_weights = torch.load('{}onelabel.pth'.format(labelroot))['labels_weights'].float().cuda().permute(1, 0)
            elif ssl_type == 'randomlabels':
                labels_weights = torch.load('{}randomlabels.pth'.format(labelroot))['labels_weights'].float().cuda().permute(1, 0)
            self.save_dict = {
                'semseg': torch.nonzero(labels_weights[0]).squeeze(),
                'depth': torch.nonzero(labels_weights[1]).squeeze(),
                'normals': torch.nonzero(labels_weights[2]).squeeze()
            }
            for k, v in self.save_dict.items():
                print(k, v.shape[0])
        elif p.train_db_name == 'Cityscapes':
            labelroot = './data/ssl_mapping/cityscapes/'
            labels_weights = torch.load('{}onelabel.pth'.format(labelroot))['labels_weights'].float().cuda().permute(1, 0)
            self.save_dict = {
                'semseg': torch.nonzero(labels_weights[0]).squeeze(),
                'depth': torch.nonzero(labels_weights[1]).squeeze()
            }
            for k, v in self.save_dict.items():
                print(k, v.shape[0])

        if p.train_db_name == 'PASCALContext':
            self.task_thresholds = {'semseg': 0.9, 'human_parts': 0.85, 'sal': 0.7, 'normals': 0.7, 'edge': 0.99}
        elif p.train_db_name == 'NYUD':
            if ssl_type == 'onelabel':
                self.task_thresholds = {'semseg': 0.9, 'depth': 0.85, 'normals': 0.85}
            elif ssl_type == 'randomlabels':
                self.task_thresholds = {'semseg': 0.9, 'depth': 0.95, 'normals': 0.85}
        elif p.train_db_name == 'Cityscapes':
            self.task_thresholds = {'semseg': 0.95, 'depth': 0.95}

        self.semiloss_dict = {
            'semseg': Spatial_CrossEntropyLoss(ignore_index=p.ignore_index, threshold=self.task_thresholds['semseg']),
        }

        if 'depth' in self.tasks:
            self.semiloss_dict['depth'] = Spatial_L1Loss(normalize=False, ignore_index=p.ignore_index,
                                          threshold=self.task_thresholds['depth'])

        if 'normals' in self.tasks:
            self.semiloss_dict['normals'] = Spatial_L1Loss(normalize=True, ignore_index=p.ignore_index,
                                      threshold=self.task_thresholds['normals'])

        if 'human_parts' in self.tasks:
            self.semiloss_dict['human_parts'] = Spatial_CrossEntropyLoss(ignore_index=p.ignore_index,
                                                    threshold=self.task_thresholds['human_parts'])

        if 'sal' in self.tasks:
            self.semiloss_dict['sal'] = Spatial_CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index,
                                            threshold=self.task_thresholds['sal'])

        if 'edge' in self.tasks:
            self.semiloss_dict['edge'] = Spatial_BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index,
                                                           threshold=self.task_thresholds['edge'])

        self.cls_loss = CrossEntropyLoss(ignore_index=p.ignore_index)


    def produce_pseudo_label(self, x, regression_logits, task):
        if task in ['semseg', 'human_parts']:
            label = nn.Softmax(dim=1)(x).max(1)[1]
            score = nn.Softmax(dim=1)(x).max(1)[0].unsqueeze(1)
        elif task in ['sal']:
            label = F.softmax(x, dim=1)[:, 1, :, :]
            score = nn.Softmax(dim=1)(x).max(1)[0].unsqueeze(1)
        elif task in ['normals']:
            label = x
            score = nn.Softmax(dim=1)(regression_logits['normals'] * 5).max(1)[0].unsqueeze(1)
        elif task in ['depth']:
            label = x
            score = nn.Softmax(dim=1)(regression_logits['depth'] * 5).max(1)[0].unsqueeze(1)
        elif task in ['edge']:
            soft_label = 1 / (1 + torch.exp(-x))
            label = (soft_label > 0.99).to(dtype=torch.float32)
            # label = soft_label
            score = torch.abs(soft_label - 0.5) * 2

        return label.detach().clone(), score.detach().clone()

    def _get_oh_label(self, reg_label, task='depth'):
        if task == 'depth':
            mask = (reg_label == 255)
            oh = torch.abs(reg_label - self.depth_set)  # B * CN * H * W
            oh_max = torch.argmin(oh, dim=1)
            oh_max = torch.where(mask == True, 255, oh_max)
        elif task == 'normals':
            mask = (reg_label == 255).all(dim=1, keepdim=True)
            oh = torch.sum(reg_label.unsqueeze(1) * self.normal_set, dim=2)  # B * CN * H * W
            oh_max = torch.argmax(oh, dim=1)
            oh_max = torch.where(mask == True, 255, oh_max)
        return oh_max

    def cal_cls_reg_loss(self, pred_score, gt, task):
        oh_label = self._get_oh_label(gt, task)
        cls_loss = self.cls_loss(pred_score, oh_label) * 0.1
        return cls_loss

    def feat_distill_loss(self, pred_feat, target_feat, feat_score):
        l2_loss = F.mse_loss(pred_feat, target_feat.detach().clone(), reduction='none')
        return (l2_loss * feat_score.detach().clone()).mean()

    def feat_loss(self, pred_feat, target_feat):
        l2_loss = F.mse_loss(pred_feat, target_feat.detach().clone(), reduction='mean')
        return l2_loss

    def slice_scores(self, reg_scores, idx):
        if reg_scores == {}:
            return {}
        else:
            return {task: reg_scores[task][idx:idx + 1] for task in reg_scores.keys()}

    def forward(self, pred, gt, pred_reg_scores, semi_utils, tasks, is_extra_data):
        out = {task: 0 for task in tasks}
        out['total'] = 0
        out['semi'] = {task: 0 for task in tasks}
        out['feat'] = {task: 0 for task in tasks}
        out['feat_s'] = {task: 0 for task in tasks}
        out['cls'] = {'depth': 0, 'normals': 0}

        if semi_utils != None:
            pred_t, pred_feat, feat_t, feat_score_t, regression_scores_t = semi_utils

        # for paritial labels available
        local_bs = gt['semseg'].shape[0]
        for idx in range(local_bs):
            img_name = gt['meta']['img_name'][idx]
            for t_idx, task in enumerate(tasks):
                if img_name in self.save_dict[task]:
                    cur_loss = self.loss_ft[task](pred[task][idx:idx + 1], gt[task][idx:idx + 1]) / local_bs * len(tasks)

                    if task in ['depth', 'normals']:
                        cls_loss = self.cal_cls_reg_loss(self.slice_scores(pred_reg_scores, idx)[task],
                                                          gt[task][idx:idx + 1], task) / local_bs * len(tasks)
                        cur_loss += cls_loss
                        out['cls'][task] += cls_loss.detach()

                    if semi_utils != None:
                        f_loss = self.feat_loss(pred_feat[task][idx:idx + 1], feat_t[task][idx:idx + 1]) / local_bs * len(tasks)
                        out['feat_s'][task] += f_loss.detach()
                        out[task] += cur_loss.detach()
                        out['total'] += cur_loss * self.loss_weights[task] + f_loss
                    else:
                        out[task] += cur_loss.detach()
                        out['total'] += cur_loss * self.loss_weights[task]

                else:
                    cur_loss = self.loss_ft[task](pred[task][idx:idx + 1], gt[task][idx:idx + 1]) / local_bs * len(tasks)
                    beta = 0.05 if task == 'edge' else 1
                    if semi_utils != None:
                        plabel, score = self.produce_pseudo_label(pred_t[task][idx:idx + 1], self.slice_scores(regression_scores_t, idx), task)
                        semi_p_loss = self.semiloss_dict[task](pred[task][idx:idx + 1], plabel, score) / local_bs * len(tasks)
                        semi_f_loss = self.feat_distill_loss(pred_feat[task][idx:idx + 1], feat_t[task][idx:idx + 1],
                                                             feat_score_t[task][idx:idx + 1]) / local_bs * len(tasks)
                        out['semi'][task] += semi_p_loss.detach()
                        out['feat'][task] += semi_f_loss.detach()
                        out['total'] += cur_loss * 0 + semi_p_loss * self.loss_weights[task] * beta + semi_f_loss
                    else:
                        out['total'] += cur_loss * 0

        return out