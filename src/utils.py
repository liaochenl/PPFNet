import numpy as np
import torch
import random
from model import DeepONet, PPONet1D

def set_seed(seed):
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    
    
def test_l2re(loader, model):
    with torch.no_grad():
        model.cpu().eval()
        train_p, train_v, y, train_u = next(iter(loader))
        # train_p, train_v, y, train_u = train_p.cuda(), train_v.cuda(), y.cuda(), train_u.cuda()
        if model.__class__.__name__ == 'DeepONet':
            pred_train = model(train_v, y)
        elif model.__class__.__name__ == 'PPONet1D':
            pred_train = model(train_p, train_v, y)
        error = torch.sqrt(torch.sum((pred_train - train_u)**2, dim=[1,2]))
        l2_true = torch.sqrt(torch.sum(train_u**2, dim=[1,2]))
        l2re = torch.mean(error/l2_true)
        pred_train = pred_train.detach().numpy()
        train_u = train_u.detach().numpy()
        erev = error / l2_true
        return l2re.detach().numpy(), train_p.detach().numpy()[...,0], erev.detach().numpy(), y.detach().numpy()[...,0], pred_train, train_u
    
def test_l2re_simply(loader, y, model):
    with torch.no_grad():
        model.cpu().eval()
        train_p, train_v, train_u = next(iter(loader))
        if model.__class__.__name__ == 'DeepONets':
            pred_train = model(train_v, y)
        elif model.__class__.__name__ == 'PPONets':
            pred_train = model(train_p, train_v, y)
        error = torch.sqrt(torch.sum((pred_train - train_u)**2, dim=[1,2]))
        l2_true = torch.sqrt(torch.sum(train_u**2, dim=[1,2]))
        l2re = torch.mean(error/l2_true)
        pred_train = pred_train.detach().numpy()
        train_u = train_u.detach().numpy()
        erev = error / l2_true
        return l2re.detach().numpy(), train_p.detach().numpy(), erev.detach().numpy(), y.detach().numpy()[...,0], pred_train, train_u
    
    
def test_aee(p, error,step=2):
    ep = p.flatten()[::step][::-1]
    er = error.flatten()[::step][::-1]
    print(f'P: {ep}')
    print(f' Error: {er}')
    aee = np.mean((er[1:]-er[:-1])/(ep[:-1]-ep[1:]))
    print(f'AEE: {aee:.2f}')
    return aee