import sys 
sys.path.append("..") 
from BFA.quantization import quantize
import torch
from torch import nn
from BFA.quantization import quan_Conv2d, quan_Linear
import torch.nn.functional as F

def SimpleOptQuanModel(model):
    '''重设定权重步长'''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()
            
    '''优化步长'''
    step_param = [
        param for name, param in model.named_parameters() if 'step_size' in name
    ]
    optimizer_quan = torch.optim.SGD(step_param,lr=0.01,momentum=0.9,weight_decay=0, nesterov=True)

    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            for i in range(300):  # runs 200 iterations to reduce quantization error
                optimizer_quan.zero_grad()
                weight_quan = quantize(m.weight, m.step_size, m.half_lvls) * m.step_size
                loss_quan = F.mse_loss(weight_quan,
                                           m.weight,
                                           reduction='mean')
                loss_quan.backward()
                optimizer_quan.step()   
    return model

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)
    
def Fp2QuModel(model):
    for n,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            weight=m.weight.data.detach()
            if m.bias!=None:
                bias_statue=True
            else:
                bias_statue=False
            new_layer=quan_Conv2d(in_channels=m.in_channels,out_channels=m.out_channels,kernel_size=m.kernel_size,\
                stride=m.stride,padding=m.padding,dilation=m.dilation,groups=m.groups,bias=bias_statue)
            new_layer.weight.data=weight
            _set_module(model, n, new_layer)
            
        elif isinstance(m, nn.Linear):
            weight=m.weight.data.detach()
            if m.bias!=None:
                bias_statue=True
            else:
                bias_statue=False
            new_layer=quan_Linear(in_features=m.in_features, out_features=m.out_features, bias=bias_statue)
            new_layer.weight.data=weight
            _set_module(model, n, new_layer)
    return model

