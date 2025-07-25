import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math


# standard MLP for DeepONet branch
class MLP_std(nn.Module):
    def __init__(self,in_features, hidden_features, num_hidden_layers, fourier=False):
        super().__init__()
        self.fourier = fourier
        if fourier:
            in_features = in_features*2        
        self.linear_in = nn.Linear(in_features,hidden_features)
        self.activation = torch.tanh
        self.layers = nn.ModuleList([self.linear_in] + [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)  ])
        
    def forward(self,x):
        if self.fourier:
            x = torch.cat([torch.cos(2*math.pi*x),torch.sin(2*math.pi*x)],dim=-1)
        for layer in self.layers:
            x = self.activation(layer(x))
    
        return x

class MLP_res(nn.Module):
    def __init__(self, in_features, hidden_features, num_hidden_layers, fourier=False):
        super().__init__()
        self.fourier = fourier
        if fourier:
            in_features = in_features * 2
        self.linear_in = nn.Linear(in_features, hidden_features)
        self.activation = torch.tanh
        self.layers = nn.ModuleList(
            [self.linear_in] + 
            [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)]
        )
        
    def forward(self, x):
        if self.fourier:
            x = torch.cat([torch.cos(2 * math.pi * x), 
                          torch.sin(2 * math.pi * x)], dim=-1)
        
        # 处理输入层
        x = self.activation(self.layers[0](x))
        
        # 处理带残差的隐藏层
        for layer in self.layers[1:]:
            residual = x
            x = layer(x)
            x += residual
            x = self.activation(x)
        
        return x   
    
class DeepONet(nn.Module):
    def __init__(self,in_b, in_t, hidden_features,num_hidden_layers):
        super().__init__()
        self.branch = MLP_res(in_b,hidden_features,num_hidden_layers)
        self.trunk = MLP_res(in_t,hidden_features,num_hidden_layers)
    def forward(self,v, y):
        
        return torch.sum(self.trunk(y)*self.branch(v),dim = -1,keepdim = True)

class DeepONets(nn.Module):
    def __init__(self,in_b, in_t, hidden_features,num_hidden_layers):
        super().__init__()
        self.branch = MLP_std(in_b,hidden_features,num_hidden_layers)
        self.trunk = MLP_std(in_t,hidden_features,num_hidden_layers)
    def forward(self,v, y):
        u = torch.einsum('ij,kj->ik',self.branch(v),self.trunk(y))
        return u.unsqueeze(-1)
    
class PPONet1D(nn.Module):
    def __init__(self,in_vb, in_t, latent_features, num_hidden_layers, p_branch=None):
        super(PPONet1D, self).__init__()
        if p_branch is None:
            # PGONet
            self.p_branch = MLP_res(1,latent_features,num_hidden_layers, fourier=True)
        else:
            # PPONet
            self.p_branch = p_branch
            # self.nn = MLP_std(latent_features,latent_features,2)
        self.v_branch = MLP_res(in_vb,latent_features,num_hidden_layers)
        self.trunk = MLP_res(in_t,latent_features,num_hidden_layers)

        self.fc_3 = nn.Linear(latent_features,1, bias = False)
        # 1024
        self.fc_1 = nn.Sequential(nn.Linear(latent_features,2*latent_features),nn.ReLU(),
                                  nn.Linear(2*latent_features,latent_features))
        
        self.fc_2 = nn.Sequential(nn.Linear(latent_features,2*latent_features),nn.ReLU(),
                                  nn.Linear(2*latent_features,latent_features))
        # self.ln = nn.LayerNorm(latent_features)

    def forward(self,p,v,y):
        v_features = self.v_branch(v)
        p_features = self.p_branch(p)
        # if hasattr(self,'nn'):
        #     p_features = self.nn(p_features)
        y_features = self.trunk(y)
        
        v_features = self.fc_1(v_features*p_features)
        y_features = self.fc_2(y_features*p_features)
        u = self.fc_3(v_features*y_features)
        
        return u
    
class PPONets(nn.Module):
    def __init__(self,in_vb, in_t, latent_features, num_hidden_layers, p_branch=None):
        super(PPONets, self).__init__()
        if p_branch is None:
            # PGONet
            self.p_branch = MLP_std(1,latent_features,num_hidden_layers, fourier=True)
        else:
            # PPONet
            self.p_branch = p_branch
        self.v_branch = MLP_std(in_vb,latent_features,num_hidden_layers)
        self.trunk = MLP_std(in_t,latent_features,num_hidden_layers)

        self.fc_3 = nn.Linear(latent_features,1, bias = False)

        self.fc_1 = nn.Sequential(nn.Linear(latent_features,1024),nn.ReLU(),
                                  nn.Linear(1024,latent_features))
        
        self.fc_2 = nn.Sequential(nn.Linear(latent_features,1024),nn.ReLU(),
                                  nn.Linear(1024,latent_features))

    def forward(self,p,v,y):
        v_features = self.v_branch(v)
        p_features = self.p_branch(p)

        y_features = self.trunk(y)
        
        v_features = self.fc_1(v_features*p_features)
        v_features = v_features.unsqueeze(1)
        y_features = torch.einsum('ij,kj->ikj',p_features,y_features)
        y_features = self.fc_2(y_features)
        u = self.fc_3(v_features*y_features)
        
        return u
    
    
class PUPRE(nn.Module): # Parameter-U Pretrain Reconstruction Encoder
    def __init__(
        self,
        temperature,
        p_latent_dim,
        u_latent_dim,
        p_encoder,
        u_encoder,
        projection_dim,
        u_length=100
    ):
        super(PUPRE,self).__init__()
        self.p_encoder = p_encoder
        self.u_encoder = u_encoder
        self.p_projection = ProjectionHead(p_latent_dim, projection_dim)
        self.u_projection = ProjectionHead(u_latent_dim, projection_dim)
        self.temperature = temperature
        self.ln = nn.LayerNorm(u_length)
        self.decoder = nn.Sequential(nn.Linear(p_latent_dim, p_latent_dim//2),nn.Tanh(),nn.Dropout(0.5),
                                    nn.Linear(p_latent_dim//2, p_latent_dim//2),nn.Tanh(), nn.Dropout(0.5),
                                    nn.Linear(p_latent_dim//2, 2))

    def forward(self, p, u, p_re):

        un = self.ln(u)
        p_features = self.p_encoder(p)
        u_features = self.u_encoder(un)
        
        p_features_re = self.p_encoder(p_re)
        pr = self.decoder(p_features_re)
        p_re = torch.cat([torch.cos(2*math.pi*p_re),torch.sin(2*math.pi*p_re)],dim=-1)
        re_loss = F.mse_loss(pr, p_re)

        p_embeddings = self.p_projection(p_features)
        u_embeddings = self.u_projection(u_features)
        
        # Calculating the Loss
        logits = (p_embeddings @ u_embeddings.T) / self.temperature
        
        p_similarity = (p_embeddings @ p_embeddings.T) / self.temperature
        u_similarity = (u_embeddings @ u_embeddings.T) /self.temperature
        # targets = F.softmax((p_similarity + u_similarity) / 2 * self.temperature, dim=-1)
        targets = torch.eye(u_similarity.shape[0]).to(u_similarity.device)
        p_loss = cross_entropy(logits, targets, reduction='none')
        u_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (p_loss+u_loss)/2 # shape: (batch_size)
        # print(loss.data)
        return 0.7*loss.mean()+0.3*re_loss
    
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
class ProjectionHead(nn.Module):
    def __init__(self,latent_features,projection_dim):
        super().__init__()
        self.projection = nn.Linear(latent_features,projection_dim)
        self.activation = torch.tanh
    def forward(self,x):
        return self.activation(self.projection(x))