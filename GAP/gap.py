import torch
import torch.nn as nn
import torch.optim as optim
from .generators import ResnetGenerator,weights_init
import numpy as np

class GAPSample():
    def __init__(self,model,mean_arr,stddev_arr):
        self.model=model
        self.generator=ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0])
        self.generator.apply(weights_init)
        self.optimizerGAP=optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion_pre = nn.CrossEntropyLoss().cuda()
        self.mean_arr=mean_arr
        self.stddev_arr=stddev_arr
        self.mag_in=10
        self.he=torch.tensor([1.0]).cuda()
    
    def normalize_and_scale(self,delta_im,bs=128):
        # from pyinstrument import Profiler
        # profiler = Profiler()
        # profiler.start()
        delta_im = delta_im + 1 # now 0..2
        delta_im = delta_im * 0.5 # now 0..1
        for c in range(3):
            delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - self.mean_arr[c]) / self.stddev_arr[c]
        for i in range(bs):
            for ci in range(3):
                l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
                mag_in_scaled_c = self.mag_in/(255.0*self.stddev_arr[ci])
                # delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())
                delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * torch.min(self.he,mag_in_scaled_c / l_inf_channel)
        # profiler.stop()
        # profiler.print()
        return delta_im
    
    def train_batch(self,images):
        # from pyinstrument import Profiler
        # profiler = Profiler()
        # profiler.start()
        
        images=images.cuda()
        self.generator.train()
        # 选择最不可能的标签
        pretrained_label_float = self.model(images)
        _, target_label = torch.min(pretrained_label_float, 1)
        
        delta_im = self.generator(images)
        delta_im = self.normalize_and_scale(delta_im,images.size(0))
        # self.generator.zero_grad()
        self.optimizerGAP.zero_grad()
        recons = torch.add(images.cuda(), delta_im.cuda())
        recons =recons.cuda()
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(images[:,cii,:,:].min(), images[:,cii,:,:].max())
        output_pretrained = self.model(recons.cuda())
        loss = torch.log(self.criterion_pre(output_pretrained, target_label))
        loss.backward()
        self.optimizerGAP.step()
        
        # profiler.stop()
        # profiler.print()
        
    def attack(self,images):
        self.generator.train().eval()
        delta_im  = self.generator(images)
        delta_im =self.normalize_and_scale(delta_im,images.size(0))
        recons  = torch.add(images, delta_im[0:images.size(0)])
        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(images[:,cii,:,:].min(), images[:,cii,:,:].max())
        attacked_images=recons
        outputs=self.model(attacked_images)
        pred=torch.nn.functional.softmax(outputs,dim=1)
        confidence,pred_label=pred.max(dim=1)
        return attacked_images,pred_label,confidence