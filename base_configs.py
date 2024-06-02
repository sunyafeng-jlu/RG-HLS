def get_base_config(args):
    args.lr_g = 1e-3
    args.lr_z = 1e-3
    args.reset_l0 = 0
    args.reset_bn = 0
    args.bn_mmt = 0
    args.is_maml = 1

    # Basic
    args.lr = 0.1
    args.T = 1
    args.epochs = 200
    args.kd_steps = 400
    args.ep_steps = 400
    args.warmup = 0

    args.evaluate_only = False
    args.batch_size = 128
    args.synthesis_batch_size = None
    # Device
    args.gpu = 0
    args.world_size = -1
    args.rank = -1
    args.dist_url = None
    args.dist_backend = 'nccl'
    args.multiprocessing_distributed = False
    args.fp16 = False

    # Misc
    args.seed = None
    args.workers = 4
    args.start_epoch = 0
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.print_freq = 0
    args.pretrained  = False
    return args

def get_config(args):
    args=get_base_config(args)
    args.g_steps=10
#    args.g_steps=50
    if args.dataset=='imagenet':
        args.batch_size =64 
        args.g_steps=50
    else:
        args.batch_size =256 
    if args.dataset=='tiny_imagenet':
        args.g_steps=50
    args.lr= 0.2 
    args.kd_steps= 400 
    args.ep_steps= 400
    args.bn =10.0 
    args.gpu= 0 
    args.T= 20 
    args.seed =None
    args.bn_mmt =0.9 
    args.warmup =20 
    args.epochs =420
    args.is_maml =1 
    args.reset_l0= 1 
    args.reset_bn =0
    args.cmi_init=None
    args.lr_z =0.01 
    args.lr_g =2e-3 
#    args.lr_g =2e-2
    args.adv =1.33
    args.oh =0.5
    if args.teacher=='resnet34':
        args.adv =1.33
        args.oh =0.5
    elif args.teacher=='vgg11':
        args.adv =1.0
        args.oh =0.5
    elif args.teacher=='wrn40_2':
        args.adv =1.1
        args.oh =0.4
    args.momentum=0.9
    args.weight_decay=1e-4
    return args