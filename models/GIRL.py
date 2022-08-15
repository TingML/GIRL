import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torchvision
import copy
import numpy as np
from torch.nn import Conv2d

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Zkl(nn.Module):
    def __init__(self):
        super(Zkl, self).__init__()
        self.conv1 = Conv2d(in_channels=312, out_channels=312, kernel_size=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        return x


class PrtAttLayer(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU(inplace=True)

    def prt_interact(self,sem_prt):
        tgt2 = self.self_attn(sem_prt, sem_prt, value=sem_prt)[0]
        sem_prt = sem_prt + self.dropout1(tgt2)
        return sem_prt
        
    def prt_assign(self, vis_prt, vis_query):
        vis_prt = self.multihead_attn(query=vis_prt,
                                   key=vis_query,
                                   value=vis_query)[0]
        return vis_prt
        
    def prt_refine(self, vis_prt):
        new_vis_prt = self.linear2(self.activation(self.linear1(vis_prt)))
        return new_vis_prt + vis_prt

    def forward(self, vis_prt, vis_query):
        # sem_prt: 196*bs*c
        # vis_query: wh*bs*c
        vis_prt = self.prt_assign(vis_prt,vis_query)
        vis_prt = self.prt_refine(vis_prt)
        return vis_prt

class PrtClsLayer(nn.Module):
    def __init__(self, nc, na, dim):
        super().__init__()
        
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        self.fc1 = nn.Linear(dim, dim//na) 
        self.fc2 = nn.Linear(dim//na, dim)
        
        self.weight_bias = nn.Parameter(torch.empty(nc, dim))
        nn.init.kaiming_uniform_(self.weight_bias, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.empty(nc))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_bias)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.activation = nn.ReLU()

    def prt_refine(self, prt):
        w = F.sigmoid(self.fc2(self.activation(self.fc1(prt))))
        prt = self.linear2(self.activation(self.linear1(prt)))
        prt = self.weight_bias + prt * w
        return prt

    def forward(self,query,cls_prt):
        cls_prt = self.prt_refine(cls_prt)
        logit = F.linear(query, cls_prt, self.bias)
        return logit,cls_prt

class Sup_att_ConLoss_clear(nn.Module):
    def __init__(self, temperature=10):
        super(Sup_att_ConLoss_clear, self).__init__()
        self.temperature = temperature

        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, features, features1,labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features1.t()),
            self.temperature)
        anchor_dot_contrast = F.normalize(anchor_dot_contrast, dim=1)

        # anchor_dot_contrast = self.similarity_f(features.unsqueeze(1), features.unsqueeze(0)) / self.temperature
        # normalize the logits for numerical stability
        # logits_max = torch.sum(anchor_dot_contrast, dim=1,keepdim=True)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits =  torch.div(anchor_dot_contrast , logits_max.detach())
        # logits = torch.where(torch.isnan(logits), torch.full_like(logits, 0),logits)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask1 = mask * logits_mask
        single_samples = (mask1.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        loss = - mean_log_prob_pos*(single_samples)

        loss = loss.sum()/(loss.shape[0])

        return loss

class GIRL(nn.Module):
    def __init__(self, pretrained=True, args=None):
        super(GIRL, self).__init__()
        ''' default '''
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size       
        vis_prt_dim = sf_size           
        self.sf =  torch.from_numpy(args.sf)
        self.args  = args
        vis_emb_dim = args.vis_emb_dim
        att_emb_dim = args.att_emb_dim
        self.args.hidden_dim=2048
        ''' backbone net'''
        if args.backbone=='resnet101':
            self.backbone = torchvision.models.resnet101()
        elif args.backbone=='resnet50':
            self.backbone = torchvision.models.resnet50()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False
        ''' ZSR '''                                      
        self.vis_proj = nn.Sequential(
            nn.Conv2d(2048, vis_emb_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(vis_emb_dim),
            nn.ReLU(inplace=True),
        )
        emb_dim = att_emb_dim*vis_prt_dim                            
        # att prt
        self.zsl_prt_emb = nn.Embedding(vis_prt_dim, vis_emb_dim)
        self.zsl_prt_dec = PrtAttLayer(dim=vis_emb_dim, nhead=8)
        self.zsl_prt_s2v = nn.Sequential(nn.Linear(vis_emb_dim,att_emb_dim),nn.LeakyReLU())

        # self.relationnet = Zkl()

        # cate prt
        self.cls_prt_emb = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.kaiming_uniform_(self.cls_prt_emb, a=math.sqrt(5))
        self.cls_prt_dec = PrtClsLayer(nc=num_classes, na=att_emb_dim, dim=emb_dim)
        # sem proj
        self.sem_proj = nn.Sequential(
            nn.Linear(sf_size,int(emb_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(emb_dim/2),emb_dim),
            nn.LeakyReLU(),
        )
        # self.proj1 = nn.Sequential(
        #     nn.Linear(512,num_classes),
        #     nn.LeakyReLU(),
        # )

        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrained:
            if args.backbone=='resnet101':
                self.backbone.load_state_dict(torch.load(args.resnet_pretrain))
            elif args.backbone=='resnet50':
                self.backbone.load_state_dict(torch.load(args.resnet_pretrain))            
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.att_contras_criterion = Sup_att_ConLoss_clear(0.5)


    def forward(self, x):

        last_conv = self.backbone(x)
        bs, c, h, w = last_conv.shape
        vis_query = self.vis_proj(last_conv).flatten(2).permute(2, 0, 1) # wh*bs*c
        sem_emb = self.sem_proj(self.sf.cuda())
        sem_emb_norm = F.normalize(sem_emb, p=2, dim=1)

        vis_prt = self.zsl_prt_emb.weight.unsqueeze(1).repeat(1, bs, 1).cuda()
        vis_prt1 = self.zsl_prt_dec(vis_prt, vis_query)
        vis_emb = self.zsl_prt_s2v(vis_prt1).permute(1,0,2).flatten(1)
        vis_emb_norm = F.normalize(vis_emb, p=2, dim=1)
        logit_zsl= vis_emb_norm.mm(sem_emb_norm.permute(1,0))

        cls_prt = self.cls_prt_emb.cuda()
        logit_cls,cls_prt = self.cls_prt_dec(vis_emb,cls_prt)

        return logit_zsl, logit_cls, None



