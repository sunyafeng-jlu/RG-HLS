import copy
from GAP.generators import ResnetGenerator, weights_init
import datafree
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv
from datafree.utils import ImagePool, DataIter, clip_images
from torchvision import transforms
from kornia import augmentation
import time

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67


def reset_l0(model):
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class RGHLSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 init_dataset=None, iterations=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0.0, bn=1, oh=1,
                 save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, device='cpu', distributed=False, lr_z = 0.01,
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,is_maml=1,
                 mode='anew',hard_loss=False):
        super(RGHLSynthesizer, self).__init__(teacher, student)
        
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.bn_mmt = bn_mmt
        self.ismaml = is_maml

        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.init_dataset = init_dataset
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.generator = generator.to(device)
        self.device = device
        self.hooks = []

        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None
        
        self.mode=mode
        self.generator_loss=[]
        
        self.copy_teacher=copy.deepcopy(teacher)
        self.copy_teacher=self.copy_teacher.to(device)
        self.copy_teacher.eval()
        
        self.hard_loss=hard_loss
        if self.hard_loss:
            self.gap=ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0]).cuda()
            self.gap.apply(weights_init)
            self.optimizer_gap=optim.Adam(self.gap.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.criterion_gap = nn.CrossEntropyLoss().cuda()
            
        if self.mode in ['anew','reuse_common']:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        elif self.mode=='reuse_fix':
            self.meta_optimizer = torch.optim.Adam(self.generator.gan1.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        elif self.mode=='reuse_finetune':
            self.meta_optimizer = torch.optim.Adam([
                {'params': self.generator.gan1.parameters()},
                {'params': self.generator.gan2.parameters(), 'lr': self.lr_g*0.1*self.iterations},
                {'params': self.generator.decoder.parameters(), 'lr': self.lr_g*0.1*self.iterations}
            ], lr=self.lr_g*self.iterations, betas=[0.5, 0.999])
            
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m, self.bn_mmt) )
        self.aug = transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])
        
    def synthesize(self, targets=None):
        start = time.time()

        self.ep+=1
        self.student.eval()
        self.teacher.eval()
        self.generator.train()
        best_cost = 1e6

        # if (self.ep == 120+self.ep_start) and self.reset_l0:
        #     if self.mode in ['reuse_fix','reuse_common','reuse_finetune']:
        #         reset_l0(self.generator.gan1)
        #     else:
        #         reset_l0(self.generator.gan1)
        #         reset_l0(self.generator.gan2)
        #         reset_l0(self.generator.decoder)
        
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)
        
        fast_generator = copy.deepcopy(self.generator)
        if self.mode in ['anew','reuse_common']:
            optimizer = torch.optim.Adam([
                {'params': fast_generator.parameters()},
                {'params': [z], 'lr': self.lr_z}
            ], lr=self.lr_g, betas=[0.5, 0.999])
        elif self.mode=='reuse_fix':
            optimizer = torch.optim.Adam([
                {'params': fast_generator.gan1.parameters()},
                {'params': [z], 'lr': self.lr_z}
            ], lr=self.lr_g, betas=[0.5, 0.999])
        elif self.mode=='reuse_finetune':
            optimizer = torch.optim.Adam([
                {'params': fast_generator.gan1.parameters()},
                {'params': fast_generator.gan2.parameters(),'lr': self.lr_g*0.1},
                {'params': fast_generator.decoder.parameters(),'lr': self.lr_g*0.1},
                {'params': [z], 'lr': self.lr_z}
            ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs) # crop and normalize
            #############################################
            # Inversion Loss
            #############################################
            t_out = self.teacher(inputs_aug)
                
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.to(self.device)

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, targets )
            if self.adv>0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean() # decision adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()

            if self.ismaml:
                if it==0: self.meta_optimizer.zero_grad()
                if self.mode in ['reuse_fix']:
                    fomaml_grad(self.generator.gan1, fast_generator.gan1)
                else:
                    fomaml_grad(self.generator.gan1, fast_generator.gan1)
                    fomaml_grad(self.generator.gan2, fast_generator.gan2)
                    fomaml_grad(self.generator.decoder, fast_generator.decoder)
                if it == (self.iterations-1): self.meta_optimizer.step()

            optimizer.step()
            self.generator_loss.append(loss.item())
            
            #############################################
            # GAP 对抗样本攻击
            #############################################
            if self.hard_loss:
                self.train_gap_batch(inputs_aug.detach())

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            if self.mode in ['reuse_fix']:
                reptile_grad(self.generator.gan1, fast_generator.gan1)
            else:
                reptile_grad(self.generator.gan1, fast_generator.gan1)
                reptile_grad(self.generator.gan2, fast_generator.gan2)
                reptile_grad(self.generator.decoder, fast_generator.decoder)
            self.meta_optimizer.step()

        self.student.train()
        self.prev_z = (z, targets)
        end = time.time()

        self.data_pool.add( best_inputs )
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {"synthetic": best_inputs}, end - start
        
    def sample(self):
        return self.data_iter.next()
  
    def train_gap_batch(self,images):
        self.gap.train()
        pretrained_label_float = self.copy_teacher(images)
        _, target_label = torch.min(pretrained_label_float, 1)
        
        delta_im = self.gap(images)
        delta_im = self.normalizer.__call__(delta_im,reverse=True)
        self.optimizer_gap.zero_grad()
        recons = torch.add(images.cuda(), delta_im.cuda())
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(images[:,cii,:,:].min(), images[:,cii,:,:].max())
        output_pretrained = self.copy_teacher(recons.cuda())
        loss_gap = torch.log(self.criterion_gap(output_pretrained, target_label))
        loss_gap.backward()
        self.optimizer_gap.step()
    
    def attack(self,images):
        self.gap.eval()
        delta_im  = self.gap(images)
        delta_im = self.normalizer.__call__(delta_im,reverse=True)
        recons  = torch.add(images, delta_im[0:images.size(0)])
        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(images[:,cii,:,:].min(), images[:,cii,:,:].max())
        attacked_images=self.aug(recons)
        outputs=self.copy_teacher(attacked_images)
        pred=torch.nn.functional.softmax(outputs,dim=1)
        confidence,pred_label=pred.max(dim=1)
        return attacked_images,pred_label,confidence,outputs

    def output_reuse_gan(self):
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        copy_generator=copy.deepcopy(self.generator)
        fake1=copy_generator.gan1(z)
        fake1=copy_generator.decoder((fake1+torch.rand((fake1.size())).cuda())/2)
        fake2=copy_generator.gan2(z)
        fake2=copy_generator.decoder((fake2+torch.rand((fake2.size())).cuda())/2)

        self.copy_teacher.eval()
        _,act1=self.copy_teacher(fake1,return_features=True)
        _,act2=self.copy_teacher(fake2,return_features=True)
        if act1.abs().sum()>act2.abs().sum():
            return copy_generator.gan2,copy_generator.gan1
        else:
            return copy_generator.gan1,copy_generator.gan2