#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F



class EmbeddedMlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_dims,
                 out_features,
                 feature_dim,
                 act_layer=nn.GELU):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_dims),
            nn.LayerNorm(hidden_dims),
            act_layer()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            act_layer()
        )
        self.fc3 = nn.Linear(hidden_dims, out_features)

        self.embedding_layers = nn.Linear(out_features, feature_dim)

        self.indicators = torch.eye(out_features).cuda()

    def forward(self, x):
        x = x + self.embedding_layers(self.indicators.unsqueeze(0)).permute(0, 2, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class TokenAttention(nn.Module):
    def __init__(self, dim, atten_dim, reduction_ratio=[4, 4, 4]):
        super(TokenAttention, self).__init__()

        self.qkv_projection = nn.ModuleList([
            nn.Linear(dim, atten_dim),
            nn.Linear(dim, atten_dim),
            nn.Linear(dim, atten_dim),
        ])
        self.spatial_reduction = nn.ModuleList([
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=reduction_ratio[0], stride=reduction_ratio[0], bias=False, groups=dim),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=reduction_ratio[1], stride=reduction_ratio[1], bias=False, groups=dim),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=reduction_ratio[2], stride=reduction_ratio[2], bias=False, groups=dim),
        ])
        self.reduction_ratio = reduction_ratio[0]
        self.scale = atten_dim ** -0.5

        self.channel_restore = nn.Linear(atten_dim, dim)
        self.token_restore = nn.Linear(atten_dim, dim)

        self.FFN = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, feature, token):

        B, C, H, W = feature.shape
        h, w = H // self.reduction_ratio, W // self.reduction_ratio

        token = token.repeat(B, 1, 1)
        q = self.qkv_projection[0](torch.cat([self.spatial_reduction[0](feature).reshape(B, C, -1).contiguous(), token], dim=-1).permute(0, 2, 1))      # (B, N, C)
        k = self.qkv_projection[1](torch.cat([self.spatial_reduction[1](feature).reshape(B, C, -1).contiguous(), token], dim=-1).permute(0, 2, 1))
        v = self.qkv_projection[2](torch.cat([self.spatial_reduction[2](feature).reshape(B, C, -1).contiguous(), token], dim=-1).permute(0, 2, 1))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2)

        updated_feature, updated_token = x[:, :, :-1], x[:, :, -1:]

        updated_feature = self.channel_restore(updated_feature.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, h, w).contiguous()
        updated_feature = F.interpolate(updated_feature, size=(H, W), mode='bilinear') + feature
        updated_feature = self.FFN(updated_feature)

        updated_token = self.token_restore(updated_token.mean(0, keepdim=True).permute(0, 2, 1)).permute(0, 2, 1)

        return updated_feature, updated_token


class TokenAttentionv2(nn.Module):
    def __init__(self, dim, atten_dim, reduction_ratio=[4, 4, 4]):
        super(TokenAttentionv2, self).__init__()

        self.qkv_projection = nn.ModuleList([
            nn.Linear(dim, atten_dim),
            nn.Linear(dim, atten_dim),
            nn.Linear(dim, atten_dim),
        ])
        self.spatial_reduction = nn.ModuleList([
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=reduction_ratio[0], stride=reduction_ratio[0], bias=False, groups=dim),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=reduction_ratio[1], stride=reduction_ratio[1], bias=False, groups=dim),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=reduction_ratio[2], stride=reduction_ratio[2], bias=False, groups=dim),
        ])
        self.reduction_ratio = reduction_ratio[0]
        self.scale = atten_dim ** -0.5

        self.channel_restore = nn.Linear(atten_dim, dim)
        self.token_restore = nn.Linear(atten_dim, dim)

        self.FFN = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.norm_feat = nn.LayerNorm(dim)
        self.norm_token = nn.LayerNorm(dim)

    def forward(self, feature, token):

        B, C, H, W = feature.shape
        h, w = H // self.reduction_ratio, W // self.reduction_ratio

        _token = token.repeat(B, 1, 1)   # B, C, 1
        q = self.qkv_projection[0](torch.cat([self.spatial_reduction[0](feature).reshape(B, C, -1).contiguous(), _token], dim=-1).permute(0, 2, 1))      # (B, N, C)
        k = self.qkv_projection[1](torch.cat([self.spatial_reduction[1](feature).reshape(B, C, -1).contiguous(), _token], dim=-1).permute(0, 2, 1))
        v = self.qkv_projection[2](torch.cat([self.spatial_reduction[2](feature).reshape(B, C, -1).contiguous(), _token], dim=-1).permute(0, 2, 1))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2)

        updated_feature, updated_token = x[:, :, :-1], x[:, :, -1:]

        updated_feature = self.channel_restore(updated_feature.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, h, w).contiguous()
        updated_feature = F.interpolate(updated_feature, size=(H, W), mode='bilinear') + feature
        updated_feature = self.FFN(updated_feature)

        updated_token = self.token_restore(updated_token.mean(0, keepdim=True).permute(0, 2, 1)).permute(0, 2, 1)

        updated_feature = feature + self.norm_feat(updated_feature.reshape(B, C, -1).contiguous().permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        updated_token = token + self.norm_token(updated_token.permute(0, 2, 1)).permute(0, 2, 1)

        return updated_feature, updated_token


class TaskSepcificPrompt(nn.Module):
    def __init__(self, dim, out_dim, reduction_ratio=2):
        super(TaskSepcificPrompt, self).__init__()

        self.task_spatial_reduction = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=reduction_ratio, stride=reduction_ratio, groups=dim)
        self.out_dim = out_dim

        self.sep_token_generator = EmbeddedMlp(out_dim, 128, out_dim, feature_dim=dim)
        self.reduction_ratio = reduction_ratio
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.FFN = nn.Conv2d(in_channels=dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        self.refine = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1)

    def vis_token(self, token):
        channel_token = self.sep_token_generator(token).squeeze()     # C Cp
        return channel_token

    def forward(self, feature, token):
        sup_feature = feature
        feat_score = self.sigmoid((sup_feature * token.unsqueeze(-1)).sum(1, keepdim=True))

        d_feature = self.task_spatial_reduction(feature).unsqueeze(0)                                       # 1  B C H W
        channel_token = self.sep_token_generator(token).permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)    # Cp 1 C 1 1

        task_score = (d_feature * channel_token).sum(2).permute(1, 0, 2, 3)          # (1 B C H W) * (Cp 1 C 1 1) = (Cp B 1 H W)
        task_score = F.interpolate(task_score, scale_factor=self.reduction_ratio, mode='bilinear')
        sep_task_score = self.softplus(task_score)

        task_pred = self.FFN(feature) * sep_task_score
        task_pred = self.refine(task_pred)

        return task_pred, sup_feature, feat_score


class DeepLabHead(nn.Module):
    def __init__(self, p, in_channels, num_classes, token_class):
        super(DeepLabHead, self).__init__()

        self.aspp = ASPP(p, in_channels, [12, 24, 36])
        self.conv = nn.Conv2d(p.TOKEN_DIM, p.TOKEN_DIM, 3, padding=1, bias=False)

        self.task_token_attention = TokenAttention(dim=p.TOKEN_DIM, atten_dim=p.TOKEN_DIM, reduction_ratio=[2, 2, 2])
        # self.task_token_attention = TokenAttentionv2(dim=p.TOKEN_DIM, atten_dim=p.TOKEN_DIM, reduction_ratio=[2, 2, 2])
        self.pred_prompter = TaskSepcificPrompt(dim=p.TOKEN_DIM, out_dim=token_class, reduction_ratio=1)

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
        x = self.aspp(x)
        feat = self.conv(x)
        task_feat, task_token = self.task_token_attention(feat, task_token)
        task_pred, task_feat, feat_score = self.pred_prompter(task_feat, task_token)
        return [task_pred, task_feat, feat_score]


class DeepLabHeadv2(nn.Module):
    def __init__(self, p, in_channels, num_classes, token_class):
        super(DeepLabHeadv2, self).__init__()

        self.aspp = ASPP(p, in_channels, [12, 24, 36])
        self.conv = nn.Conv2d(p.TOKEN_DIM, p.TOKEN_DIM, 3, padding=1, bias=False)

        task_token_attention_list = [TokenAttentionv2(dim=p.TOKEN_DIM, atten_dim=p.TOKEN_DIM, reduction_ratio=[2, 2, 2]) for _ in range(p.NUM_HEAD_LAYER)]
        self.task_token_attention_list = nn.ModuleList(task_token_attention_list)
        self.pred_prompter = TaskSepcificPrompt(dim=p.TOKEN_DIM, out_dim=token_class, reduction_ratio=1)

        self.num_head_layer = p.NUM_HEAD_LAYER

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
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, task_token):
        x = self.aspp(x)
        feat = self.conv(x)
        token = task_token
        for i in range(self.num_head_layer):
            feat, token = self.task_token_attention_list[i](feat, token)
        task_pred, task_feat, feat_score = self.pred_prompter(feat, token)
        return [task_pred, task_feat, feat_score]


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, p, in_channels, atrous_rates):
        super(ASPP, self).__init__()

        out_channels = p.TOKEN_DIM
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):

        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
