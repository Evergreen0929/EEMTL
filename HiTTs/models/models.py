#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from models.normal_clusters import normal_clu_center, depth_linear_clu_center, depth_linear_clu_center_citys
from thop import profile

class TaskTokenPrompt(nn.Module):
    def __init__(self, dim, reduction_ratio=2, task_num=3):
        super(TaskTokenPrompt, self).__init__()

        self.task_spatial_reduction = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=reduction_ratio, stride=reduction_ratio, groups=dim)
        self.reduction_ratio = reduction_ratio
        self.scale = dim ** -0.5
        self.task_num = task_num
        self.sigmoid = nn.Sigmoid()

        self.qk_producer = nn.ModuleList([
            nn.Linear(dim, dim),
            nn.Linear(dim, dim)
        ])

        FFN = []
        for i in range(task_num):
            FFN.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU()
                )
            )
        self.FFN = nn.ModuleList(FFN)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _decompose(self, x, expand):
        items = []
        for item in x:
            if expand:
                item = item.unsqueeze(0)
            items.append(item)
        return items

    def _group_FFN_forward(self, features):
        out = []
        for i in range(self.task_num):
            out.append(self.FFN[i](features[i]))
        return out

    def _cross_task_affinity(self, tokens):
        tokens = torch.cat(tokens, dim=-1).permute(0, 2, 1)
        q = self.qk_producer[0](tokens)
        k = self.qk_producer[1](tokens)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return attn

    def _cross_task_reweight_feature(self, attn, features):
        B, C, H, W = features[0].shape
        attn = attn.repeat(B, 1, 1)
        _features = []
        for feat in features:
            _features.append(feat.reshape(B, -1).contiguous().unsqueeze(0))
        _features = torch.cat(_features, dim=0).permute(1, 0, 2)

        out = (attn @ _features).permute(1, 0, 2).reshape(self.task_num, B, C, H, W).contiguous()
        out_feat = self._decompose(out, expand=False)

        return out_feat

    def _cross_task_reweight_token(self, attn, tokens):
        tokens = torch.cat(tokens, dim=-1).permute(0, 2, 1)      # 1, 3, C
        out = (attn @ tokens).permute(1, 2, 0)                   # 3, C, 1
        out_token = self._decompose(out, expand=True)

        return out_token

    def forward(self, backbone_feature, task_tokens):
        d_back_features = self.task_spatial_reduction(backbone_feature)

        task_features = []
        for token in task_tokens:
            attn_score = self.sigmoid((d_back_features * token.unsqueeze(-1)).sum(1, keepdim=True))
            attn_score = F.interpolate(attn_score, scale_factor=self.reduction_ratio, mode='bilinear')
            task_features.append(backbone_feature * attn_score)

        cross_affinity_map = self._cross_task_affinity(task_tokens)
        re_task_features = self._cross_task_reweight_feature(cross_affinity_map, task_features)
        re_task_tokens = self._cross_task_reweight_token(cross_affinity_map, task_tokens)

        out_task_features = self._group_FFN_forward(re_task_features)

        return out_task_features, re_task_tokens


class MultiscaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiscaleFusion, self).__init__()

        # yhr: to combine
        last_inp_channels = sum(in_channels)
        self.combine_conv = nn.Conv2d(last_inp_channels, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        # x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear', align_corners=False)
        # x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear', align_corners=False)
        # x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear', align_corners=False)
        # x = torch.cat([x[0], x1, x2, x3], 1)

        rescaled_x = [F.interpolate(_x, (x0_h, x0_w), mode='bilinear', align_corners=False) for _x in x]
        x = torch.cat(rescaled_x, 1)
        x = self.combine_conv(x)
        return x


class MultiscaleTokenPrompt(nn.Module):
    def __init__(self, in_channels, out_channels, out_token_dim, task_num):
        super(MultiscaleTokenPrompt, self).__init__()

        self.multi_scale_task_prompter = nn.ModuleList([TaskTokenPrompt(dim=dim, reduction_ratio=1, task_num=task_num) for dim in in_channels])
        self.token_projector = nn.ModuleList([nn.ModuleList([nn.Linear(in_channels[-1], dim) for dim in in_channels]) for _ in range(task_num)])
        self.token_restore = nn.ModuleList([nn.ModuleList([nn.Linear(dim, out_token_dim) for dim in in_channels]) for _ in range(task_num)])
        self.stage = len(in_channels)
        self.task_num = task_num

        # yhr: to combine
        last_inp_channels = sum(in_channels)
        self.combine_conv = nn.ModuleList([nn.Conv2d(last_inp_channels, out_channels, kernel_size=1) for _ in range(task_num)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, task_token):
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        out_feat_list = []
        out_token_list = []
        ms_task_tokens = []
        for i in range(self.stage):
            ms_task_tokens.append([self.token_projector[j][i](task_token[j].permute(0, 2, 1)).permute(0, 2, 1) for j in range(self.task_num)])  # ti: [1, C, 1] * 4

        ms_prompt_output = [self.multi_scale_task_prompter[i](x[i], ms_task_tokens[i]) for i in range(self.stage)]

        for j in range(self.task_num):
            x1 = F.interpolate(ms_prompt_output[1][0][j], (x0_h, x0_w), mode='bilinear', align_corners=False)
            x2 = F.interpolate(ms_prompt_output[2][0][j], (x0_h, x0_w), mode='bilinear', align_corners=False)
            x3 = F.interpolate(ms_prompt_output[3][0][j], (x0_h, x0_w), mode='bilinear', align_corners=False)
            x = torch.cat([ms_prompt_output[0][0][j], x1, x2, x3], 1)
            out_feat_list.append(self.combine_conv[j](x))
            out_token_list.append(sum([self.token_restore[j][i](ms_prompt_output[i][1][j].permute(0, 2, 1)).permute(0, 2, 1) for i in range(self.stage)]))

        return out_feat_list, out_token_list


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.task = task

    def forward(self, x):
        out_size = x.size()[2:]
        out = self.decoder(self.backbone(x))
        return {self.task: F.interpolate(out, out_size, mode='bilinear', align_corners=False)}


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        return {task: [F.interpolate(self.decoders[task](shared_representation)[0], out_size, mode='bilinear'),
                       F.interpolate(self.decoders[task](shared_representation)[1], out_size, mode='bilinear')] for task in self.tasks}


class MultiTaskPrompter(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, p, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list, feat_dim: list):
        super(MultiTaskPrompter, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

        self.tokens = nn.Parameter(torch.randn(len(tasks), feat_dim[-1], 1)).cuda()
        self.token_reduction = nn.ModuleList([nn.Linear(feat_dim[-1], p.TOKEN_DIM) for _ in range(len(tasks))])
        self.fusion = MultiscaleFusion(feat_dim, feat_dim[-1])
        self.task_prompter = TaskTokenPrompt(dim=feat_dim[-1], reduction_ratio=1, task_num=len(tasks))

        if 'normals' in self.tasks:
            self.normal_set = normal_clu_center().cuda()

        if 'depth' in self.tasks:
            if len(self.tasks) == 2:
                self.depth_set = depth_linear_clu_center_citys(50).cuda()
            else:
                self.depth_set = depth_linear_clu_center(30).cuda()

    def produce_normal_token_pred(self, x):
        x = nn.Softmax(dim=1)(x)
        normal_pred = torch.sum(x.unsqueeze(2) * self.normal_set.cuda(), dim = 1)
        normal_pred = normal_pred / torch.norm(normal_pred, p=2, dim=1, keepdim=True)

        return normal_pred

    def produce_depth_token_pred(self, x):
        x = nn.Softmax(dim=1)(x)
        depth_pred = torch.sum(x * self.depth_set.cuda(), dim = 1, keepdim=True)

        return depth_pred

    def forward(self, batch):
        x = batch['image']  # .cuda(non_blocking=True)
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        prepared_token_list = [self.tokens[i].unsqueeze(0) for i in range(len(self.tasks))]

        init_task_feat_list, init_token_list = self.task_prompter(self.fusion(shared_representation), prepared_token_list)

        d_token_list = [self.token_reduction[i](init_token_list[i].permute(0, 2, 1)).permute(0, 2, 1) for i in range(len(self.tasks))]

        final_list = {task: self.decoders[task](init_task_feat_list[i], d_token_list[i]) for i, task in enumerate(self.tasks)}

        regression_scores = {}
        if 'normals' in self.tasks:
            regression_scores['normals'] = final_list['normals'][0]
            final_list['normals'][0] = self.produce_normal_token_pred(final_list['normals'][0])
        if 'depth' in self.tasks:
            regression_scores['depth'] = final_list['depth'][0]
            final_list['depth'][0] = self.produce_depth_token_pred(final_list['depth'][0])

        return ({task: F.interpolate(final_list[task][0], out_size, mode='bilinear') for task in self.tasks},
                {task: final_list[task][1] for task in self.tasks},
                {task: final_list[task][2] for task in self.tasks},
                {task: F.interpolate(regression_scores[task], out_size, mode='bilinear') for task in
                 regression_scores.keys()} if regression_scores != {} else {})


class MTMSPrompter(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, p, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list, feat_dim: list):
        super(MTMSPrompter, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

        self.tokens = nn.Parameter(torch.randn(len(tasks), feat_dim[-1], 1)).cuda()
        self.multi_scale_task_prompter = MultiscaleTokenPrompt(in_channels=feat_dim, out_channels=feat_dim[-1],
                                                               out_token_dim=p.TOKEN_DIM, task_num=len(self.tasks))

        if 'normals' in self.tasks:
            self.normal_set = normal_clu_center().cuda()

        if 'depth' in self.tasks:
            if len(self.tasks) == 2:
                self.depth_set = depth_linear_clu_center_citys(50).cuda()
            else:
                self.depth_set = depth_linear_clu_center(30).cuda()

    def produce_normal_token_pred(self, x):
        x = nn.Softmax(dim=1)(x)
        normal_pred = torch.sum(x.unsqueeze(2) * self.normal_set.cuda(), dim = 1)
        normal_pred = normal_pred / torch.norm(normal_pred, p=2, dim=1, keepdim=True)

        return normal_pred

    def produce_depth_token_pred(self, x):
        x = nn.Softmax(dim=1)(x)
        depth_pred = torch.sum(x * self.depth_set.cuda(), dim = 1, keepdim=True)

        return depth_pred

    def forward(self, batch):
        x = batch['image']  # .cuda(non_blocking=True)
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        prepared_token_list = [self.tokens[i].unsqueeze(0) for i in range(len(self.tasks))]

        init_task_feat_list, init_token_list = self.multi_scale_task_prompter(shared_representation, prepared_token_list)

        final_list = {task: self.decoders[task](init_task_feat_list[i], init_token_list[i]) for i, task in enumerate(self.tasks)}

        regression_scores = {}
        if 'normals' in self.tasks:
            regression_scores['normals'] = final_list['normals'][0]
            final_list['normals'][0] = self.produce_normal_token_pred(final_list['normals'][0])
        if 'depth' in self.tasks:
            regression_scores['depth'] = final_list['depth'][0]
            final_list['depth'][0] = self.produce_depth_token_pred(final_list['depth'][0])

        return ({task: F.interpolate(final_list[task][0], out_size, mode='bilinear') for task in self.tasks},
                {task: final_list[task][1] for task in self.tasks},
                {task: final_list[task][2] for task in self.tasks},
                {task: F.interpolate(regression_scores[task], out_size, mode='bilinear') for task in
                 regression_scores.keys()} if regression_scores != {} else {})

