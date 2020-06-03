import torchreidHash
import torch
import numpy as np
import os
from torchreidHash.utils import load_pretrained_weights
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
#load_pretrained_weights(model,'log/model_duke.pth')
# model = torchreid.models.build_model(
#     name='resnet50',
#     num_classes=datamanager.num_train_pids,
#     loss='triplet',
#     pretrained=True
# )
model = model.cuda()




#from torchsummary import summary
# summary(model,(3,160,2048))



optimizer = torchreidHash.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreidHash.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)
engine = torchreidHash.engine.ImageTripletEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)
engine.run(
    save_dir='log/myModel_new',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False,
    #visrank=True,  
)



