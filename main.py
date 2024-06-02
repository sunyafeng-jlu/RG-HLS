import argparse
import os
import random
import time
import warnings

import numpy
from GAP.gap import GAPSample
from base_configs import get_config

from datafree.utils.logger import get_beijin_time, print_log
from datafree.utils.reuse import get_generator, get_synthesizer, prepare_model

import registry
import datafree
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

parser = argparse.ArgumentParser(description='Reusable Generator Data-free Knowledge Distillation with adversarial example attack-based hard loss strategy')

# reuse
parser.add_argument('--mode', default='anew', choices=['anew', 'reuse_common','reuse_finetune','reuse_fix'])
parser.add_argument('--reuse_generator_path', default='/wxw2/syf/projects/RGHL_DFKD/run/anew-cifar10-resnet34-resnet18/gan_reuse.pth',type=str)
parser.add_argument('--reuse_decoder_path', default='/wxw2/syf/projects/RGHL_DFKD/run/anew-cifar10-resnet34-resnet18/decoder.pth',type=str)

# hard loss
parser.add_argument('--hard_loss', action='store_true')
parser.add_argument('--alpha', default=0.9, type=float)
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--no_aea', action='store_true')
parser.add_argument('--pa', default=0.001, type=float)

# basic
parser.add_argument('--data_root', default='/wxw/datasets/')
parser.add_argument('--teacher', default='resnet50_imagenet')
parser.add_argument('--student', default='resnet18_imagenet')
parser.add_argument('--dataset', default='imagenet')
parser.add_argument('--save_dir', default='run/', type=str)
parser.add_argument('--save_synthetic_images', default=True,type=bool)
parser.add_argument('--save_train_samples', default=False,type=bool)
parser.add_argument('--test_every_epoch', default=True,type=bool)

best_acc1 = 0
time_cost = 0

def main():
    args = parser.parse_args()
    args = get_config(args)
        
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node= torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global time_cost
    args.gpu = gpu
    # GPU and FP16
    if args.gpu is not None:
        print("Use GPU: {} for training:".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx

    # Logger
    if args.hard_loss:
        if args.no_aea:
            args.save_dir=args.save_dir+args.mode+'-hl-noaea-'+args.dataset+'-'+args.teacher+'-'+args.student+'/'
        else:
            args.save_dir=args.save_dir+args.mode+'-hl-'+args.dataset+'-'+args.teacher+'-'+args.student+'/'
    else:
        args.save_dir=args.save_dir+args.mode+'-'+args.dataset+'-'+args.teacher+'-'+args.student+'/'
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    log_name='log-'+get_beijin_time()+'.txt'
    log = open(os.path.join(args.save_dir,log_name), 'w')
    print_log('==>show run configs',log)
    if args.rank<=0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            print_log( "   %s: %s"%(k,v),log)

    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)
    print_log('==>loading dataset success',log)

    # Setup models
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    if args.dataset!='imagenet':
        teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu')['state_dict'])
    args.normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    student = prepare_model(args,student)
    teacher = prepare_model(args,teacher)
    print_log('==>loading teacher and student success',log)

    # Setup the data-free synthesizer
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size
    generator=get_generator(args)
    synthesizer,criterion=get_synthesizer(args,teacher,student,generator,num_classes,ori_dataset)
    print_log('==>loading generator and synthesizer success',log)

    # Setup optimizer
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warmup, eta_min=2e-4)
    print_log('==>set optimizer finish',log)

    # Test models
    print_log('==>evaluate teacher and student',log)
    eval_results = evaluator(teacher, device=args.gpu)
    (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
    print_log('   [teacher] Acc@1={:.4f} Acc@5={:.4f} Loss={:.4f}'.format(acc1,acc5,val_loss),log)
    eval_results = evaluator(student, device=args.gpu)
    (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
    print_log('   [student] Acc@1={:.4f} Acc@5={:.4f} Loss={:.4f}'.format(acc1,acc5,val_loss),log)
    
    # Train Loop
    print_log('==>starting train:',log)
    for epoch in range(args.start_epoch, args.epochs):
        st=time.time()
        args.current_epoch=epoch
        for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
            vis_results, cost = synthesizer.synthesize() # g_steps
            time_cost += cost
            if epoch >= args.warmup:
                train( synthesizer, [student, teacher], criterion, optimizer, args, log) # kd_steps 
        if args.save_synthetic_images:
            for _, vis_image in vis_results.items():
                datafree.utils.save_image_batch( vis_image, args.save_dir+'synthetic.png')
                torch.save(vis_image,args.save_dir+'synthetic.pth')
                
                if args.hard_loss and not args.no_aea:
                    torch.save(synthesizer.gap,args.save_dir+'gap.pth')
                    
        student.eval() 
        if args.test_every_epoch: 
            eval_results = evaluator(student, device=args.gpu)
            (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        else:
            acc1, acc5, val_loss = -1,-1,-1
        et=time.time()
        print_log('   Epoch {}: Acc@1={:.4f} Acc@5={:.4f} Student Loss={:.4f} Lr={:.4f} Generator Loss={:.4f} Cost Time={:.2f}'.format(\
            args.current_epoch,acc1, acc5, val_loss, optimizer.param_groups[0]['lr'],synthesizer.generator_loss[-1],et-st),log)
            
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = args.save_dir+'best.pth'
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)

        if epoch >= args.warmup:
            scheduler.step()

        # save
        # gan_reuse,gan_abandon=synthesizer.output_reuse_gan()
        gan_reuse,gan_abandon=synthesizer.generator.gan1,synthesizer.generator.gan2
        torch.save(gan_abandon,args.save_dir+'gan_abandon.pth')
        torch.save(gan_reuse,args.save_dir+'gan_reuse.pth')
        torch.save(synthesizer.generator.decoder,args.save_dir+'decoder.pth')
        numpy.save(args.save_dir+'generator_loss.npy', synthesizer.generator_loss)
                
    if args.rank<=0:
        print_log("==>Best: %.4f"%best_acc1,log)
        print_log("==>Generation Cost: %1.3f" % (time_cost/3600.),log)
        log.close()

# do the distillation
def train(synthesizer, model, criterion, optimizer, args,log):
    global time_cost
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()

    for i in range(args.kd_steps):
        images = synthesizer.sample()
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
                
        if args.hard_loss:
            if args.no_aea:
                pred=torch.nn.functional.softmax(t_out,dim=1)
                _,labels=pred.max(dim=1)
            else:
                images,t_out,labels=identity_swap_probabilistic_attack(t_out,images,synthesizer,args.synthesis_batch_size,args.pa)
            
        with args.autocast():       
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())
            
            if args.hard_loss:
                loss_hard=torch.nn.functional.cross_entropy(s_out, labels)
                loss_s = loss_s*args.alpha + loss_hard*args.beta
             
        optimizer.zero_grad()
        if args.fp16:
            scaler = args.scaler
            scaler.scale(loss_s).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        
        if args.print_freq == -1 and i % 10 == 0 and args.current_epoch >= 150:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            print_log(
                '[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, \
                    train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps, train_acc1=train_acc1,
                        train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']),log)
            loss_metric.reset(), acc_metric.reset()
        elif args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            print_log('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, \
                train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps, train_acc1=train_acc1, \
                  train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']),log)
            loss_metric.reset(), acc_metric.reset()
    
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)
 
def identity_swap_probabilistic_attack(t_out,images,synthesizer,synthesis_batch_size,pa):     
    # fake样本和被攻击的fake样本根据置信度身份互换
    pred=torch.nn.functional.softmax(t_out,dim=1)
    confidence,labels=pred.max(dim=1)
    attacked_images,attacked_labels,attacked_confidence,attacked_out=synthesizer.attack(images.detach())
    # ind_ato=attacked_confidence>=confidence
    ind_ato= (attacked_confidence>=confidence) & (attacked_labels==labels)
    if ind_ato.sum().item()!=0:
        temp1,temp2=images.detach(),t_out.detach()
        images[ind_ato],labels[ind_ato]=attacked_images[ind_ato],attacked_labels[ind_ato]
        attacked_images[ind_ato]=temp1[ind_ato]
        t_out[ind_ato]=attacked_out[ind_ato]
        attacked_out[ind_ato]=temp2[ind_ato]

    # 以一定概率被攻击
    ind_att=torch.rand((synthesis_batch_size))<=pa
    if ind_att.sum().item()!=0:
        images[ind_att]=attacked_images[ind_att]
        t_out[ind_att]=attacked_out[ind_att]
        
    return images,t_out,labels

if __name__ == '__main__':
    main()