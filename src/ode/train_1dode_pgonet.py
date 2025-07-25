import sys
sys.path.append('../../src/')
import numpy as np
import torch
from model import PPONet1D, PUPRE, MLP_std
from dataset import get_data, get_dataloader
from utils import set_seed, test_l2re, test_aee


if __name__ == "__main__":

    # Set seed (0,10,100)
    seed = 0
    set_seed(seed)
    num_samples = 100
    test_rho = 0.1
    data_type = 'ODE'
    pre = False
    if pre:
        print('PPONet loading!!!')
         ## PreTrain Process: PUPRE 
        pretrain_epoch = 10001
        # Load data
        p, u, pr = get_data(data_type, task_type='PUPRE', test_rho=test_rho, num_samples=num_samples)
        p = p.cuda()
        u = u.cuda()
        pr = pr.cuda()
        # define the model
        temperature = 0.8
        p_latent_dim = 50
        u_latent_dim = 50
        p_encoder = MLP_std(1, p_latent_dim,4, fourier=True)
        u_encoder = MLP_std(100,u_latent_dim,4)
        projection_dim = 50
        net = PUPRE(temperature, p_latent_dim, u_latent_dim,
                p_encoder, u_encoder, projection_dim, u_length=100)
        net.cuda()
        # define the optimizer
        opt1 = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        print(f'Start Pretrain PUPRE: total {pretrain_epoch} epochs')
        for epoch in range(pretrain_epoch):
            net.train()
            opt1.zero_grad()
            loss = net(p, u, pr)
            loss.backward()
            opt1.step()
            if epoch % 100 == 0:
                print(epoch, loss.item())
        print('Pretrain PUPRE Done!')
        p_branch = net.p_encoder
    else:
        print('PGONet loading!!!')
        p_branch = None
        
    ## Train PPONet
    hidden_dim = 50
    num_layers = 4
    in_vb, in_t = 101, 2
    train_epoch = 20001
    # Load data
    train_loader, val_loader, test_l_loader, test_r_loader = get_dataloader(data_type, task_type='PPONet', test_rho=test_rho, num_works=0, num_samples=num_samples)
    
    # define the model
    model = PPONet1D(in_vb, in_t, hidden_dim, num_layers, p_branch=p_branch)
    # if pre:
    #     for param in model.p_branch.parameters():
    #         param.requires_grad = False
    model.cuda()
    # define the optimizer
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    lrs = torch.optim.lr_scheduler.StepLR(opt2, step_size=2000, gamma=0.7)
    print(f'Start Train PPONet: total {train_epoch} epochs')
    for epoch in range(train_epoch):
        model.train()
        train_p, train_v, y, train_u = next(iter(train_loader))
        train_p, train_v, y, train_u = train_p.cuda(), train_v.cuda(), y.cuda(), train_u.cuda()
        opt2.zero_grad()
        loss = torch.nn.MSELoss()(model(train_p, train_v, y), train_u)
        loss.backward()
        opt2.step()
        lrs.step()
        if epoch % 100 == 0:
            print(epoch, loss.item())
    print('Train PPONet Done!')
    
    # Test
    train_l2re = test_l2re(train_loader, model)[0]
    val_l2re = test_l2re(val_loader, model)[0]
    test_l_l2re, test_p, test_error = test_l2re(test_l_loader, model)[:3]
    test_r_l2re = test_l2re(test_r_loader, model)[0]
    print(f'Train L2RE: {train_l2re:.2E}, Val L2RE: {val_l2re:.2E}, Test L L2RE: {test_l_l2re:.2E}, Test R L2RE: {test_r_l2re:.2E}')
    test_aee(test_p, test_error)
    print('Test Done!')
    torch.save(model.state_dict(), 'PGONet_ODE.pth')
    print('Model saved!')
    


