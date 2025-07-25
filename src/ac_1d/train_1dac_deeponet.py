import sys
sys.path.append('../../src/')
import numpy as np
import torch
from model import DeepONet
from dataset import get_data, get_dataloader
from utils import set_seed, test_l2re, test_aee


if __name__ == "__main__":

    # Set seed (0,10,100)
    seed = 0
    set_seed(seed)
    num_samples = 10000
    test_rho = 0.1
    data_type = '1D_AC'
        
    ## Train DeepONet
    hidden_dim = 50
    num_layers = 4
    in_b, in_t = 129, 3
    train_epoch = 20001
    # Load data
    train_loader, val_loader, test_l_loader, test_r_loader = get_dataloader(data_type, task_type='PPONet', test_rho=test_rho, num_works=0, batch_size=500, num_samples=num_samples)
    
    # define the model
    model = DeepONet(in_b, in_t, hidden_dim, num_layers)
    model.cuda()
    # define the optimizer
    opt1 = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.StepLR(opt1, step_size=2000, gamma=0.5)
    print(f'Start Train DeepONet: total {train_epoch} epochs')
    model.train()
    for epoch in range(train_epoch):
        loss_value = 0
        for i, (_, train_v, y, train_u) in enumerate(train_loader):
            train_v, y, train_u = train_v.cuda(), y.cuda(), train_u.cuda()
            opt1.zero_grad()
            loss = torch.nn.MSELoss()(model(train_v, y), train_u)
            loss.backward()
            opt1.step()
            lrs.step()
            loss_value += loss.item()
        if epoch % 100 == 0:
            print(epoch, loss_value)
    print('Train DeepONet Done!')
    
    # Test
    train_l2re = test_l2re(train_loader, model)[0]
    val_l2re = test_l2re(val_loader, model)[0]
    test_l_l2re, test_p, test_error = test_l2re(test_l_loader, model)[:3]
    test_r_l2re = test_l2re(test_r_loader, model)[0]
    print(f'Train L2RE: {train_l2re:.2E}, Val L2RE: {val_l2re:.2E}, Test L L2RE: {test_l_l2re:.2E}, Test R L2RE: {test_r_l2re:.2E}')
    test_aee(test_p, test_error, step=200)
    print('Test Done!')
    # torch.save(model.state_dict(), 'DeepONet_1D_AC.pth')
    # print('Model saved!')
    