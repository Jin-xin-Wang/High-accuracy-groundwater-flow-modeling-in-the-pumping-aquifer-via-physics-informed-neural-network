# -*- coding: utf-8 -*-
"""
Neural Network Models - Heterogeneous Single-Well Pumping Flow PINN
Includes: PirateNet, Multi-scale Fourier Feature Networks
"""
import torch
import torch.nn as nn
import numpy as np
from utils import device, get_activation, mse_loss
import torch.nn.functional as F


# ============== Basic Layers ==============
def swish(x):
    return x * torch.sigmoid(x)

# ============== Backup Model: Standard MLP ==============
class MLP(nn.Module):
    """Standard Multi-Layer Perceptron"""
    def __init__(self, args):
        super(MLP, self).__init__()

        activation = get_activation(args.activation)

        layers = []
        layers.append(nn.Linear(args.in_dim, args.hid_layer_dim))
        layers.append(activation())

        for _ in range(args.hid_layer_num):
            layers.append(nn.Linear(args.hid_layer_dim, args.hid_layer_dim))
            layers.append(activation())

        layers.append(nn.Linear(args.hid_layer_dim, args.out_dim))

        self.net = nn.Sequential(*layers)

        # Initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)



# ============== Spatiotemporal Fourier Feature Network - Non-trainable FF ==============
class T2S2FF_one_branch_Net(nn.Module):
    """Fourier Feature Network (Spatiotemporal Separation)"""
    def __init__(self, args):
        super(T2S2FF_one_branch_Net, self).__init__()

        self.t_nonlinear = getattr(args, 't_nonlinear', True)

        activation = get_activation(args.activation)

        self.register_buffer('T_F1', torch.randn((1, 64), device=device) * 1)
        self.register_buffer('T_F2', torch.randn((1, 64), device=device) * 10)
        self.register_buffer('S_F1', torch.randn((2, 64), device=device) * 1)
        self.register_buffer('S_F2', torch.randn((2, 64), device=device) * 100)

        # Temporal network
        t_layers = []
        t_layers.append(nn.Linear(128,args.hid_layer_dim))
        t_layers.append(activation())
        for _ in range(args.hid_layer_num-1):
            t_layers.append(nn.Linear(args.hid_layer_dim, args.hid_layer_dim))
            t_layers.append(activation())
        self.densenet = nn.Sequential(*t_layers)

        # Output layer
        self.out_layer = nn.Linear(4 * args.hid_layer_dim, args.out_dim)

        # Initialization
        for net in [self.densenet]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, H):

        # Time transformation
        if self.t_nonlinear:
            # Log transformation (handles multi-scale time)
            t = torch.log10(H[..., 0:1]+1e-3) + 1.0
        else:
            # Linear
            t = H[..., 0:1]
        xy = H[..., 1:3]

        # Fourier feature encoding
        H1_t = torch.cat((torch.sin(t @ self.T_F1), torch.cos(t @ self.T_F1)), -1)
        H2_t = torch.cat((torch.sin(t @ self.T_F2), torch.cos(t @ self.T_F2)), -1)
        H1_xy = torch.cat((torch.sin(xy @ self.S_F1), torch.cos(xy @ self.S_F1)), -1)
        H2_xy = torch.cat((torch.sin(xy @ self.S_F2), torch.cos(xy @ self.S_F2)), -1)

        # Network processing
        H1_t = self.densenet(H1_t)
        H2_t = self.densenet(H2_t)
        H1_xy = self.densenet(H1_xy)
        H2_xy = self.densenet(H2_xy)

        # Feature fusion (Hadamard product)
        H1 = torch.mul(H1_t, H1_xy)
        H2 = torch.mul(H1_t, H2_xy)
        H3 = torch.mul(H2_t, H1_xy)
        H4 = torch.mul(H2_t, H2_xy)

        H = torch.cat((H1, H2, H3, H4), -1)
        output = self.out_layer(H)

        return output

# ============== PINN Physics Constraints ==============
class PINN_Physics:
    """PINN Physics Constraint Computation"""
    def __init__(self, scale_config):
        self.cfg = scale_config
        self.a1 = scale_config.a1
        self.a2 = scale_config.a2
        self.a3 = scale_config.a3
        self.a4 = scale_config.a4

    def get_pde_residual(self, x, pred):
        """
        Compute PDE residual (heterogeneous confined aquifer)
        Equation: Ss * dh/dt = d/dx(K*dh/dx) + d/dy(K*dh/dy)
        Expanded: Ss * dh/dt = K*(d²h/dx² + d²h/dy²) + dK/dx*dh/dx + dK/dy*dh/dy

        Input x: (batch, n_points, 6) - [t, x, y, K, Kx, Ky]
        Input pred: (batch, n_points, 1) - predicted head
        """
        # First-order derivatives
        u_grad = torch.autograd.grad(
            pred[..., 0:1], x,
            grad_outputs=torch.ones_like(pred[..., 0:1]),
            retain_graph=True, create_graph=True
        )[0]

        u_t = u_grad[..., 0:1]   # dh/dt
        u_x = u_grad[..., 1:2]   # dh/dx
        u_y = u_grad[..., 2:3]   # dh/dy

        # Extract K field information
        K = x[..., 3:4]
        Kx = x[..., 4:5]   # dK/dx
        Ky = x[..., 5:6]   # dK/dy

        # Second-order derivatives
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True
        )[0][..., 1:2]

        u_yy = torch.autograd.grad(
            u_y, x,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True, create_graph=True
        )[0][..., 2:3]

        # PDE residual (dimensionless)
        res = (K * self.a1 * u_xx + K * self.a2 * u_yy +
               Kx * self.a3 * u_x + Ky * self.a4 * u_y - u_t) / 500

        return (res ** 2).mean()

    def get_well_flux_loss(self, x, pred, labels, well_K):
        """
        Compute well flux constraint loss
        Constrain velocity on 0.5m radius circle around well

        Input x: (batch, n_points, 3) - [t, x, y]
        Input pred: predicted head
        Input labels: (batch, n_points, 2) - target velocity [vx, vy]
        Input well_K: hydraulic conductivity at well
        """
        u_grad = torch.autograd.grad(
            pred[..., 0:1], x,
            grad_outputs=torch.ones_like(pred[..., 0:1]),
            retain_graph=True, create_graph=True
        )[0]

        # Darcy's law: v = -K * grad(h) (dimensionless)
        h_scale = self.cfg.h_scale
        x_scale = self.cfg.x_scale
        y_scale = self.cfg.y_scale
        K_scale = self.cfg.K_scale

        # Predicted velocity
        vx_pred = -well_K * K_scale * h_scale * u_grad[..., 1:2] / x_scale
        vy_pred = -well_K * K_scale * h_scale * u_grad[..., 2:3] / y_scale

        return mse_loss(vx_pred, labels[..., 0:1]) + mse_loss(vy_pred, labels[..., 1:2])

    def get_neumann_bc_loss(self, x, pred, direction='y'):
        """
        Compute zero-flux boundary condition loss (Neumann BC)
        dh/dn = 0

        direction: 'x' or 'y'
        """
        u_grad = torch.autograd.grad(
            pred, x,
            grad_outputs=torch.ones_like(pred),
            retain_graph=True, create_graph=True
        )[0]

        if direction == 'y':
            # Top/bottom boundaries: dh/dy = 0
            grad_n = u_grad[..., 2:3]
        else:
            # Left/right boundaries: dh/dx = 0
            grad_n = u_grad[..., 1:2]

        return (grad_n ** 2).mean()/100


# ============== Model Factory ==============
def get_model(model_name, args):
    """Get model by name"""
    models = {
        'T2S2FF_one_branch_Net':T2S2FF_one_branch_Net,
        'MLP': MLP,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}, available: {list(models.keys())}")

    return models[model_name](args)
