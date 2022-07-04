import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision

import pandas as pd
import numpy as np

from tqdm import tqdm
import time
import random
from collections import Counter
import os


from albumentations import *
# from autoaugment import ImageNetPolicy, SVHNPolicy
from utils import *
from student_model import CCT_tokenizer_ViT as student_net
from teacher_model import CNN_model_dilated_conv as teacher_net
import config

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


T_net = teacher_net(n_channels=config.IN_CHANS, n_st=config.N_CLASSES, n_classes=config.N_CLASSES)
print('### Teacher model defined')

state_dict = torch.load('./best_epoch_teacher_model.pth.tar')
print(state_dict['epoch'])
print(state_dict['val_loss'])
print(state_dict['val_accuracy'])
T_net.load_state_dict(state_dict['model_state_dict'])
print('### Teacher model weights are imported')

S_net = student_net(in_chans=config.IN_CHANS, n_classes=config.N_CLASSES, embed_dim=config.EMBED_DIM, depth=config.DEPTH, n_heads=config.N_HEADS, mlp_ratio=config.MLP_RATIO, p=0, attn_p=0)
print('### Student model defined')

print()



print(f'Total Number of trainable parameters: {sum([param.numel() for param in S_net.parameters()])}')
print()

S_net.cuda()
print('Student Model pushed to cuda!!!')
T_net.cuda()
print('Teacher Model pushed to cuda!!!')
print()


train_loader = ...
val_loader = ...
#################################################################
criterion = ...
optimizer = ...

num_epochs = ...


alpha = config.ALPHA         ########## Take it from paper
T = config.TEMPERATURE       ########## Take it from paper


######################################### MOdify the training loop accordingly

for epoch in range(num_epochs):
    
    train_epoch_loss = 0
    train_epoch_acc = 0

    epoch_losses = AverageMeter()

    S_net.train()

    curr_start = time.time()


    for batchidx, (x,label) in enumerate(train_loader):

        x = x.float().cuda()
        label = label.squeeze().cuda()

        with torch.no_grad():
            teacher_logits = T_net(x)


        student_logits, distill_logits = S_net(x)   

        CE_loss = F.cross_entropy(student_logits, label)

        ##################################### Soft Distillation        
        # distill_loss = F.kl_div(
        #     F.log_softmax(distill_logits / T, dim = -1),
        #     F.softmax(teacher_logits / T, dim = -1).detach(),
        # reduction = 'batchmean')
        
        # distill_loss *= T ** 2
        ########################################################

        ##################################### Hard Distillation
        teacher_labels = teacher_logits.argmax(dim = -1)
        distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        train_loss = CE_loss * (1 - alpha) + distill_loss * alpha
        ########################################################
        
        # train_loss = criterion(logits, label)
        
        epoch_losses.update(train_loss.item(), len(x))
        
        train_acc = multi_acc(student_logits, label)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


        train_loss = train_loss.detach().cpu().numpy()
        train_epoch_loss += train_loss
        train_epoch_acc += train_acc.item()



    S_net.eval()
                                        #Test mode
    with torch.no_grad():

        val_epoch_loss = 0
        val_epoch_acc = 0

        for batchidx, (x,label) in enumerate(val_loader):

            x = x.float().cuda()
            label = label.squeeze().cuda()

            student_logits, distill_logits = S_net(x) 

            val_loss = F.cross_entropy(student_logits, label)

            val_acc = multi_acc(student_logits, label)

            val_loss = val_loss.detach().cpu().numpy()
            val_epoch_loss += val_loss
            val_epoch_acc += val_acc.item()
            
            
    curr_end = time.time()
        
    print() 
    print(f'Epoch Number {epoch + 1} time taken: {(curr_end - curr_start)//60} min')
    print()   
    print(f'Epoch {epoch+1:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
    print()




print('###' * 20)
print(f'Experiment Name ==> {config.EXPT_NAME} Done!')
print('###' * 20)












