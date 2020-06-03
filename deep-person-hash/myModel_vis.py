import torchreidHash
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

datamanager = torchreidHash.data.ImageDataManager(
    root='reid-data',
    #sources='dukemtmcreid',
    #targets='dukemtmcreid',
    #sources='cuhk03',
    #targets='cuhk03',
    sources='market1501',
    targets='market1501',
    height=384,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop','random_erase']
)


model = torchreidHash.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='triplet',
    pretrained=True
)
# model = torchreid.models.build_model(
#     name='resnet50',
#     num_classes=datamanager.num_train_pids,
#     loss='triplet',
#     pretrained=True
# )
model = model.cuda()




#from torchsummary import summary
# summary(model,(3,160,2048))


import torch
from torchviz import make_dot

x=torch.randn(1,3,384,128)
vis_graph=make_dot(model(x))
vis_graph.view()





