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
        accs_b = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        center_loss=CenterLoss(num_classes=751, feat_dim=4608)
        #center_loss_h=CenterLoss(num_classes=751, feat_dim=256)
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
            outputs, features,h, b,cls_score,b_classify= self.model(imgs)
            #print(len(logits_list))
            #print(logits_list[0].shape)	
            pids_g=self.parse_pids(pids,self.datamanager.num_train_pids)
            x=features
            pids_ap=pids_g.cuda()
            #AP_loss=aploss_criterion(b_classify,pids_ap)
            
            target_b = F.cosine_similarity(b[:pids_g.size(0) // 2], b[pids_g.size(0) // 2:])
            target_x = F.cosine_similarity(x[:pids_g.size(0) // 2], x[pids_g.size(0) // 2:])
            
            

            loss1 = F.mse_loss(target_b, target_x)
            loss2 = torch.mean(torch.abs(torch.pow(torch.abs(h) - Variable(torch.ones(h.size()).cuda()), 3)))
            loss_greedy = loss1 + 0.1 * loss2
            loss_batchhard_hash=self.compute_hashbatchhard(b,pids)
 
            #print(features.shape)
            loss_t = self._compute_loss(self.criterion_t, features, pids)#+self._compute_loss(self.criterion_t,b,pids)
            loss_x = self._compute_loss(self.criterion_x, outputs, pids)+self._compute_loss(self.criterion_x, b_classify, pids)+self._compute_loss(self.criterion_x, cls_score, pids)
            
            centerloss=0#center_loss(features,pids)#+center_loss_h(h,pids)
            centerloss=centerloss*0.0005

            #print(centerloss)
            loss =centerloss+self.weight_t * loss_t + self.weight_x * loss_x+loss_greedy+loss_batchhard_hash*2#+AP_loss
#            loss =centerloss + self.weight_x * loss_x+loss_greedy+loss_batchhard_hash*2#+AP_loss

            
            
            
            
            
            
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            #losses_t.update(loss_t.item(), pids.size(0))
            losses_t.update(loss_t.item(), pids.size(0))
            losses_x.update(loss_x.item(), pids.size(0))

            accs.update(metrics.accuracy(outputs, pids)[0].item())
            accs_b.update(metrics.accuracy(b_classify, pids)[0].item())
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
                    'Loss_cl {loss_cl:.4f} )\t'
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                    'Acc_b {acc_b.val:.2f} ({acc_b.avg:.2f})\t'
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
                        loss_cl=centerloss,
                        #loss_ap=AP_loss,
                        acc=accs,
                        acc_b=accs_b,
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
        
    def parse_pids(self,pids,num_classes):
        pids_return=torch.zeros(pids.shape[0],num_classes)
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

class APLoss (nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:

        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq-1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1
        q=q.cuda()

    def forward(self, x, label, qw=None, ret='1-mAP'):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        
        q = self.quantizer(x.unsqueeze(1))
        
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == 'AP':
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_ap': float(loss)}  


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


 
