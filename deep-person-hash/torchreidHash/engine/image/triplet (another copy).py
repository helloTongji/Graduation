from __future__ import division, print_function, absolute_import
import time
import datetime

from typing import Tuple
from torchreidHash import metrics
from torchreidHash.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
from torchreidHash.losses import TripletLoss, CrossEntropyLoss

from ..engine import Engine
from torch import nn,Tensor

import torch
import math
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable


from torch.nn.modules import loss

class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss(num_classes=751)
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = triplet_loss(outputs[:1], labels)
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        # print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
        #     loss_sum.data.cpu().numpy(),
        #     Triplet_Loss.data.cpu().numpy(),
        #     CrossEntropy_Loss.data.cpu().numpy()),
        #       end=' ')
        return loss_sum

class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=751,
                 m=0.3,
                 s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        w_norm=w_norm.cuda()
        costh = torch.mm(x_norm, w_norm)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss



def to_scalar(vt):
  """Transform a length-1 pytorch Variable or Tensor to scalar. 
  Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]), 
  then npx = tx.cpu().numpy() has shape (1,), not 1."""
  if isinstance(vt, Variable):
    return vt.data.cpu().numpy().flatten()[0]
  if torch.is_tensor(vt):
    return vt.cpu().numpy().flatten()[0]
  raise TypeError('Input should be a variable or tensor')
class ImageTripletEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageTripletEngine, self
              ).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        
    
        

        
        
    def train(
        self,
        epoch,
        max_epoch,
        writer,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None
    ):
        losses_t = AverageMeter()
        losses_x = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        loss_meter = AverageMeter()

        
        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(self.train_loader)
        end = time.time()
        
        
        layer_nums=3
        for batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)

            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            self.optimizer.zero_grad()
            outputs, features,h, b,y_resnet,mgn_1,mgn_2,mgn_3= self.model(imgs)
            #print(len(logits_list))
            #print(logits_list[0].shape)	
            pids_g=self.parse_pids(pids)
            x=features

            
            target_b = F.cosine_similarity(b[:pids_g.size(0) // 2], b[pids_g.size(0) // 2:])
            target_x = F.cosine_similarity(x[:pids_g.size(0) // 2], x[pids_g.size(0) // 2:])
            
            


            loss1 = F.mse_loss(target_b, target_x)
            loss2 = torch.mean(torch.abs(torch.pow(torch.abs(h) - Variable(torch.ones(h.size()).cuda()), 3)))
            loss_greedy = loss1 + 0.1 * loss2
            loss_batchhard_hash=self.compute_hashbatchhard(b,pids)
 
            
            loss_t = self._compute_loss(self.criterion_t, features, pids)
            loss_x = self._compute_loss(self.criterion_x, outputs, pids)+self._compute_loss(self.criterion_x, y_resnet, pids)+self._compute_loss(self.criterion_x, mgn_1, pids)+self._compute_loss(self.criterion_x, mgn_2, pids)+self._compute_loss(self.criterion_x, mgn_3, pids)
            

            loss = self.weight_t * loss_t + self.weight_x * loss_x+loss_greedy+loss_batchhard_hash*2

            
            
            
            
            
            
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses_t.update(loss_t.item(), pids.size(0))
            losses_x.update(loss_x.item(), pids.size(0))

            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (
                    num_batches - (batch_idx+1) + (max_epoch -
                                                   (epoch+1)) * num_batches
                )
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                    'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                    'Loss_g {loss_g:.4f} )\t'
                    'Loss_p {loss_p:.4f} )\t'
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                    'Lr {lr:.6f}\t'
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss_t=losses_t,
                        loss_x=losses_x,
                        loss_g=loss_greedy,
                        loss_p=loss_batchhard_hash,
                        acc=accs,
                        lr=self.optimizer.param_groups[0]['lr'],
                        eta=eta_str
                    )
                )

            if writer is not None:
                n_iter = epoch*num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                writer.add_scalar('Train/Data', data_time.avg, n_iter)
                writer.add_scalar('Train/Loss_t', losses_t.avg, n_iter)
                writer.add_scalar('Train/Loss_x', losses_x.avg, n_iter)
                writer.add_scalar('Train/Acc', accs.avg, n_iter)
                writer.add_scalar(
                    'Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter
                )

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
    def calculate_cos_distance(self,a,b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        cose = torch.mm(a,b.t())
        cose=1-cose
        dist=torch.sum(cose)
        return dist
        
        
    def logcosh(self, x):
        return torch.log(torch.cosh(x)) 
        
    def parse_pids(self,pids):
        pids_return=torch.zeros(pids.shape[0],751)
        for i in range(pids.shape[0]):
            pids_return[i][pids[i]-1]=1
        
        return pids_return    
    def compute_hashbatchhard(self,hashcode,targets):
        loss_return=0
        n=hashcode.shape[0]
        len=hashcode.shape[1]
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask=mask.data.cpu().numpy()
        dist_hamming=np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_hamming[i][j]=self.hamming(hashcode[i],hashcode[j])
        for i in range(n):
            dist_ap=dist_hamming[i][mask[i]].max()
            dist_an=dist_hamming[i][mask[i]==0].min()
            
            if(dist_ap>dist_an):
                loss_return+=0.0001+dist_ap-dist_an
            elif(dist_ap!=0):
                loss_return+=0.0005+dist_ap#-dist_hamming[i][mask[i]==0].min()
            else:
                if(0.25*len>dist_an):
                    loss_return+=(0.25*len-dist_an)*0.25
                # loss_return+=np.max(10-dist_an,0)*0.5
        loss_return/=dist_hamming.shape[0]
        return loss_return
        
        
    def hamming(self,hash_a,hash_b):
        mask=(hash_b==hash_a)
        dist=hash_a.shape[0]-torch.sum(mask)
        dist=dist.data.cpu().numpy()
        return dist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def compute_kshloss(self,b,pids):
        n=pids.shape[0]
        S=torch.ones((n,n))
        for i in range(n):
            for j in range(n):
                if pids[i]!=pids[j]:
                    S[i][j]=torch.tensor(-1)
        
        S=S.to("cuda: 0")
        b=b.to("cuda: 0")
        loss_ksh=torch.tensor(0)
        loss_ksh=loss_ksh.float()
        matrix_F=torch.mm(b,torch.t(b))/b.shape[1]-S
        for i in range(matrix_F.shape[0]):
            for j in range(matrix_F.shape[1]):
                loss_ksh+=(matrix_F[i][j]**2)
        return math.sqrt(loss_ksh)*0.08
        
    def cauchy_cross_entropy(self, u, label_u, v=None, label_v=None, gamma=1, normed=True):
        if v is None:
            v = u
            label_v = label_u
        label_u=label_u.float()
        label_v=label_v.float()
        
        label_ip = torch.matmul(label_u, torch.t(label_v))
        label_ip=label_ip.float()
        
        
        s = torch.clamp(label_ip, 0.0, 1.0)

        if normed:
            ip_1 = torch.matmul(u, torch.t(v))

            def reduce_shaper(t):
                return torch.reshape(torch.sum(t, 1), [t.shape[0], 1])
            mod_1 = torch.sqrt(torch.matmul(reduce_shaper(u.pow(2)), torch.t(reduce_shaper(
                v.pow(2)) + 0.000001)))
            dist = torch.tensor(np.float32(256)) / 2.0 * \
                (1.0 - torch.div(ip_1, mod_1) + 0.000001)
        else:
            r_u = torch.reshape(torch.sum(u * u, 1), [-1, 1])
            r_v = torch.reshape(torch.sum(v * v, 1), [-1, 1])

            dist = r_u - 2 * torch.matmul(u, torch.t(v)) + \
                torch.t(r_v) + 0.001

        cauchy = gamma / (dist + gamma)

        s_t = torch.mul(torch.add(s, -0.5), 2.0)
        sum_1 = torch.sum(s)
        sum_all = torch.sum(torch.abs(s_t))
        balance_param = torch.add(
            torch.abs(torch.add(s, -1.0)), torch.mul(torch.div(sum_all, sum_1), s))

        mask = torch.equal(torch.eye(u.shape[0]), torch.tensor(0.0))

        cauchy_mask = cauchy[mask]
        s_mask = s[mask]
        balance_p_mask = balance_param[mask]

        all_loss = - s_mask * \
            torch.log(cauchy_mask) - (1.0 - s_mask) * \
            torch.log(1.0 - cauchy_mask)

        return torch.mean(torch.mul(all_loss, balance_p_mask))    
        
