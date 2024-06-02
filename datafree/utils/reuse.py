import copy
import torch
import datafree

def prepare_model(args,model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                args.batch_size = int(args.batch_size / args.ngpus_per_node)
                args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model

def get_generator(args):
    nz = 256
    if args.dataset=='tiny_imagenet':
        gan1 = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=64, nc=3)
    elif args.dataset=='imagenet':
        gan1 = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=224, nc=3)
    else:
        gan1 = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
    
    if args.mode=='anew':
        gan2=copy.deepcopy(gan1)
        decoder=Decoder()
        auto_fusion=AutoFusion(gan1,gan2,decoder)
        auto_fusion=prepare_model(args,auto_fusion)
        return auto_fusion
    else:
        gan2=torch.load(args.reuse_generator_path)
        decoder=torch.load(args.reuse_decoder_path)
        auto_fusion=AutoFusion(gan1,gan2,decoder)
        auto_fusion=prepare_model(args,auto_fusion)
        return auto_fusion
       
def get_synthesizer(args,teacher,student,generator,num_classes,ori_dataset):
    criterion = datafree.criterions.KLDiv(T=args.T)
    nz = 256
    if args.hard_loss:
        if args.no_aea:
            hl=False
        else:
            hl=True
    else:
        hl=False

    if args.dataset=='tiny_imagenet':
        isize=(3,64,64)
    elif args.dataset=='imagenet':
        isize=(3,224,224)
    else:
        isize=(3,32,32)
    synthesizer = datafree.synthesis.RGHLSynthesizer(teacher, student, generator,
                nz=nz, num_classes=num_classes, img_size=isize, init_dataset=args.cmi_init,
                save_dir=args.save_dir+'samples/', device=args.gpu,
                transform=ori_dataset.transform, normalizer=args.normalizer,
                synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                adv=args.adv, bn=args.bn, oh=args.oh,
                reset_l0=args.reset_l0, reset_bn=args.reset_bn,
                bn_mmt=args.bn_mmt, is_maml=args.is_maml,mode=args.mode,hard_loss=hl)
    return synthesizer,criterion

def addition_fusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layers = torch.nn.Sequential()
        
        '''原始版本'''
        self.layers.add_module('Conv2', torch.nn.Conv2d(3,8,3,1,1))
        self.layers.add_module('Bn2', torch.nn.BatchNorm2d(8))
        self.layers.add_module('Act2' , torch.nn.ReLU(inplace=True))
        self.layers.add_module('Conv3', torch.nn.Conv2d(8,16,3,1,1))
        self.layers.add_module('Bn3', torch.nn.BatchNorm2d(16))
        self.layers.add_module('Act3' , torch.nn.ReLU(inplace=True))        
        self.layers.add_module('Conv4', torch.nn.Conv2d(16,8,3,1,1))
        self.layers.add_module('Bn4', torch.nn.BatchNorm2d(8))
        self.layers.add_module('Act4' , torch.nn.ReLU(inplace=True))
        self.layers.add_module('Conv5', torch.nn.Conv2d(8,3,3,1,1))
        self.layers.add_module('Sig5', torch.nn.Sigmoid())
        
    def forward(self, x):
        return self.layers(x)
    
class AutoFusion(torch.nn.Module):
    def __init__(self,gan1,gan2,decoder):
        super(AutoFusion, self).__init__()
        self.gan1 = gan1
        self.gan2 = gan2
        self.decoder=decoder
    
    def forward(self,z):
        out1=self.gan1(z)
        out2=self.gan2(z) 
        out=addition_fusion(out1,out2)
        out=self.decoder(out)
        
        return out
