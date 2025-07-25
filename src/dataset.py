import sys
sys.path.append('../../src/')
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import math, copy, h5py

## datset create

def get_data(data_type, task_type, test_rho=0.1, num_samples=1000):
    home_dir = '../PFs_1D/'
    # 1D Linear ADR Equation
    if data_type == '1D_Linear_ADR':
        def f(X, epx):
            X = X.astype(np.float32)
            f1 = np.exp(- X) - np.exp(-X * (1/epx))
            f2 = np.exp(-1) - np.exp(-1/epx)
            fu = f1 / f2
            return fu
        epxs = np.linspace(0.0001, 0.1, num_samples)
        X = np.linspace(0, 1, 100)
        u = []
        for epx in epxs:
            u.append(f(X, epx))
        u = np.stack(u, axis=0).reshape(-1, 100, 1)
        y = X.reshape(1,-1,1).repeat(num_samples, axis=0)
        p = epxs.reshape(-1, 1, 1)
        v = np.array([[0,1]]).reshape(1,1,2).repeat(num_samples, axis=0)
    elif data_type == 'ODE':
        data_dir = "antiderivative_p_dataset.npz"
        data = np.load(home_dir+data_dir, allow_pickle=True)
        p = data["p"]
        v = data["v"]
        u = data["u"]
        y = data["y"]
    elif data_type == '1D_AC' or data_type == '1D_CH':
        # 1D Allen-Cahn Equation or 1D Cahn-Hilliard Equation
        data_dir = '1DAC_refer_gamma_0.0001_0.1.hdf5' if data_type == '1D_AC' else '1DCH_refer_gamma_1e-06_0.0001.hdf5'
        with h5py.File(home_dir+data_dir, 'r') as f:
            gamma = f['gamma'][:]
            t = f['t-coordinate'][:]
            u_train = f['tensor'][:]
            x = f['x-coordinate'][:]
        x_res = 16
        t_res = 4
        x_train = x[::x_res]
        t_train = t[::t_res]
        xx, tt = np.meshgrid(x_train, t_train)
        x_train = xx.flatten()[:,None]
        t_train = tt.flatten()[:,None]
        y = np.concatenate([x_train, t_train], axis=1)
        y = np.expand_dims(y, 0)
        v = np.expand_dims(u_train[:,0,:], 1)[...,::8]
        y = y.repeat(v.shape[0], axis=0)
        u = u_train[:,::t_res, ::x_res].reshape(-1,y.shape[1],1)
        p = gamma[:].reshape(-1,1,1)
    elif data_type == '2D_AC':
        # 2D Allen-Cahn Equation
        data_dir = '../PFs_2D/2DAC_refer_kappa_1e-04_0.01.hdf5'
        with h5py.File(data_dir, 'r') as f:
            gamma = f['Eps'][:]
            t = f['t-coordinate'][:]
            u_train = f['tensor'][:]
            x = f['x-coordinate'][:]
        x_res = 4
        t_res = 4
        x_train = x[::x_res]
        t_train = t[::t_res]
        xx, yy, tt = np.meshgrid(x_train, x_train, t_train)
        x_train = xx.transpose(2,0,1).flatten()[:,None]
        y_train = yy.transpose(2,0,1).flatten()[:,None]
        t_train = tt.transpose(2,0,1).flatten()[:,None]
        y = np.concatenate([x_train, y_train, t_train], axis=1)
        y = np.expand_dims(y, 0)
        v = np.expand_dims(u_train[:,0,:], 1)
        v = v[:,:,::x_res,::x_res].reshape(v.shape[0],1,-1)[...,::8]
        y = y.repeat(v.shape[0], axis=0)
        u = u_train[:,::t_res, ::x_res, ::x_res].reshape(-1,y.shape[1],1)
        p = gamma[:].reshape(-1,1,1)
        
    elif data_type == '2D_CH':
        # 2D Cahn-Hilliard Equation
        data_dir = '../PFs_2D/2DCH_refer_Eps_0.01_0.1.hdf5'
        with h5py.File(data_dir, 'r') as f:
            gamma = f['Eps'][:]
            t = f['t-coordinate'][:]
            u_train = f['tensor'][:]
            x = f['x-coordinate'][:]
        x_res = 4
        t_res = 4
        x_train = x[:-1][::x_res]
        t_train = t[::t_res]
        xx, yy, tt = np.meshgrid(x_train, x_train, t_train)
        x_train = xx.transpose(2,0,1).flatten()[:,None]
        y_train = yy.transpose(2,0,1).flatten()[:,None]
        t_train = tt.transpose(2,0,1).flatten()[:,None]
        y = np.concatenate([x_train, y_train, t_train], axis=1)
        y = np.expand_dims(y, 0)
        v = np.expand_dims(u_train[:,0,:-1,:-1], 1)
        v = v[:,:,::x_res,::x_res].reshape(v.shape[0],1,-1)[...,::8]
        y = y.repeat(v.shape[0], axis=0)
        u = u_train[...,:-1,:-1][:,::t_res, ::x_res, ::x_res].reshape(-1,y.shape[1],1)
        p = gamma[:].reshape(-1,1,1)**2        
    else:
        raise ValueError('Data type not supported')

    if task_type == 'PUPRE':
        p = p[...,0]
        u = u[...,0]
        p_len = num_samples
        test_pos_l = int(p_len*test_rho)
        test_pos_r = int(p_len*(1-test_rho))
        # p = np.concatenate([np.cos(2*math.pi*p), np.sin(2*math.pi*p)], axis=-1)
        pr = copy.deepcopy(p)
        p = p[test_pos_l:test_pos_r][::2]
        v = v[test_pos_l:test_pos_r][::2]
        u = u[test_pos_l:test_pos_r][::2]
        p = torch.tensor(p).float()
        v = torch.tensor(v).float()
        u = torch.tensor(u).float()
        # pr
        pr = torch.tensor(pr).float()
        return p, u, pr
    elif task_type == 'PPONet':
        p = torch.tensor(p).float()
        v = torch.tensor(v).float()
        u = torch.tensor(u).float()
        y = torch.tensor(y).float()
        # p = torch.sin(500000*math.pi*p)
        v = torch.cat([v, p], dim=-1)
        y = torch.cat([y, p.repeat(1,y.shape[1],1)], dim=-1)
        # pr = copy.deepcopy(p)
        # pr = torch.cat([torch.cos(2*math.pi*pr), torch.sin(2*math.pi*pr)], dim=-1)
        
        p_len = num_samples
        test_pos_l = int(p_len*test_rho)
        test_pos_r = int(p_len*(1-test_rho))
        data_y = y[:test_pos_l]
        data_v = v[:test_pos_l]
        data_u = u[:test_pos_l]
        data_p = p[:test_pos_l]
        test_l_dataset = TensorDataset(data_p, data_v, data_y, data_u)
        
        data_y = y[test_pos_l:test_pos_r][::2]
        data_v = v[test_pos_l:test_pos_r][::2]
        data_u = u[test_pos_l:test_pos_r][::2]
        data_p = p[test_pos_l:test_pos_r][::2]
        train_dataset = TensorDataset(data_p, data_v, data_y, data_u)
        
        data_y = y[test_pos_l:test_pos_r][1::2]
        data_v = v[test_pos_l:test_pos_r][1::2]
        data_u = u[test_pos_l:test_pos_r][1::2]
        data_p = p[test_pos_l:test_pos_r][1::2]
        val_dataset = TensorDataset(data_p, data_v, data_y, data_u)
        
        data_y = y[test_pos_r:]
        data_v = v[test_pos_r:]
        data_u = u[test_pos_r:]
        data_p = p[test_pos_r:]
        test_r_dataset = TensorDataset(data_p, data_v, data_y, data_u)
        return train_dataset, val_dataset, test_l_dataset, test_r_dataset
    else:
        raise ValueError('Task type not supported')



def get_dataloader(data_type, task_type='PPONet', test_rho=0.1, batch_size=1000, num_works=8, num_samples=1000):
    train_dataset, val_dataset, test_l_dataset, test_r_dataset = get_data(data_type, task_type, test_rho, num_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_works, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_l_loader = DataLoader(test_l_dataset, batch_size=len(test_l_dataset), shuffle=False)
    test_r_loader = DataLoader(test_r_dataset, batch_size=len(test_r_dataset), shuffle=False)  
    return train_loader, val_loader, test_l_loader, test_r_loader