# -*- coding: utf-8 -*-
"""
Configuration File - Heterogeneous Single-Well Pumping Flow PINN (Fixed Head Boundary)
"""
import argparse
import numpy as np


def str2bool(v):
    """Convert string to boolean (for argparse)"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Cannot parse boolean value: {v}')


def str2list(v):
    """Convert string to list (for argparse)

    Supported formats:
    - "1,100" -> [1.0, 100.0]
    - "[1,100]" -> [1.0, 100.0]
    - "1 100" -> [1.0, 100.0]
    """
    import json
    if isinstance(v, list):
        return v
    # Remove leading/trailing spaces and brackets
    v = v.strip().strip('[]')
    # Try splitting by comma or space
    if ',' in v:
        items = v.split(',')
    else:
        items = v.split()
    return [float(x.strip()) for x in items if x.strip()]


class ScaleConfig:
    """Dimensionless scaling configuration"""
    def __init__(self, h_max=6.0):
        # Scale parameters
        self.x_scale = 500.0    # m
        self.y_scale = 500.0    # m
        self.t_scale = 2.0     # d
        self.h_scale = h_max    # m
        self.K_scale = 1.0      # m/d

        # Physical parameters
        self.M = 50.0           # Aquifer thickness m
        self.Ss = 2e-5          # Specific storage
        self.r_well = 0.5       # Well constraint radius m

        # PDE dimensionless coefficients
        self.a1 = self.t_scale / (self.Ss * self.x_scale**2)
        self.a2 = self.t_scale / (self.Ss * self.y_scale**2)
        self.a3 = self.t_scale / (self.Ss * self.x_scale)
        self.a4 = self.t_scale / (self.Ss * self.y_scale)

    def normalize_t(self, t):   return t / self.t_scale
    def normalize_x(self, x):   return x / self.x_scale
    def normalize_y(self, y):   return y / self.y_scale
    def normalize_h(self, h):   return h / self.h_scale
    def normalize_K(self, K):   return K / self.K_scale
    def normalize_Kx(self, Kx): return Kx / self.K_scale
    def normalize_Ky(self, Ky): return Ky / self.K_scale
    def denormalize_h(self, h): return h * self.h_scale


class WellConfig:
    """Well configuration - Single pumping well"""
    def __init__(self):
        self.positions = np.array([
            [250.0, 250.0],  # Pumping well (center)
        ])
        self.n_wells = 1
        self.well_types = np.array([-1])  # -1: pumping



class TrainingConfig:
    """Training configuration"""
    def __init__(self, args=None, domain_cfg=None):
        # Link to domain config (t_min/t_max from DomainConfig)
        self.domain_cfg = domain_cfg if domain_cfg else DomainConfig()

        # Default values (no longer includes t_min/t_max, unified from domain_cfg)
        defaults = {
            'epochs': 300000,
            'lr': 1e-3,
            'seed': 666,
            'batch_size_coll': 4096,
            'n_time_steps': 200,
            'time_sampling': 'lg',
            'spatial_sampling': 'shifty',  # Spatial collocation sampling: uniform, constant, shifty
            'n_coll_points': 30000,  # Number of collocation points (per time step)
            'use_obse_4x4': True,
            'well_obse_radius': 0.5,
            'n_well_obse_points': 60,
            'obse_4x4_repeat': 18,
            'well_flux_radius': 0.5,
            'n_well_flux_points': 120,
            'hid_layer_dim': 256,
            'hid_layer_num': 6,
            't_nonlinear': True,
        }

        # Set attributes
        for key, val in defaults.items():
            setattr(self, key, getattr(args, key, val) if args else val)

        # Auto-calculated batch sizes
        # self.batch_size_ic = self.batch_size_coll // 4
        # self.batch_size_diri = self.batch_size_coll // 2
        # self.batch_size_neum = self.batch_size_coll // 4  # Neumann BC batch size

    @property
    def t_min(self):
        """Time lower bound (linked from DomainConfig)"""
        return self.domain_cfg.t_min

    @property
    def t_max(self):
        """Time upper bound (linked from DomainConfig)"""
        return self.domain_cfg.t_max


class DomainConfig:
    """Domain configuration - Single source of truth for spatiotemporal range"""
    def __init__(self):
        # Spatial range
        self.x_min, self.x_max = 0.0, 500.0
        self.y_min, self.y_max = 0.0, 500.0
        # Time range (single definition, referenced elsewhere)
        self.t_min, self.t_max = 2/1000, 2.0
        # Boundary condition
        # self.h_boundary = 0.0

def get_args():
    """Get command line arguments"""
    p = argparse.ArgumentParser(description='Heterogeneous Single-Well Pumping Flow PINN Training')

    # Model
    p.add_argument('--model_name', type=str, default='T2S2FF_one_branch_Net',
                   choices=['MLP','T2S2FF_one_branch_Net'])
    p.add_argument('--hid_layer_dim', type=int, default=100)
    p.add_argument('--hid_layer_num', type=int, default=5, help='Number of hidden layers + 1')
    p.add_argument('--out_dim', type=int, default=1)
    p.add_argument('--t_nonlinear', type=str2bool, default=False,
                   help='Whether to use nonlinear time transformation (true/false)')
    p.add_argument('--activation', type=str, default='sin')
    p.add_argument('--Four_T', type=str2list, default=[1, 10],
                   help='Temporal Fourier feature frequency range, e.g. "0.01,1"')
    p.add_argument('--Four_S', type=str2list, default=[1, 100],
                   help='Spatial Fourier feature frequency range, e.g. "1,100"')

    # Training
    p.add_argument('--epochs', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--batch_size_coll', type=int, default=65536)

    # Data
    p.add_argument('--n_time_steps', type=int, default=200)
    p.add_argument('--time_sampling', type=str, default='lg',
                   choices=['uniform', 'lg'])
    p.add_argument('--spatial_sampling', type=str, default='shifty',
                   choices=['uniform', 'constant', 'shifty'],
                   help='Spatial collocation sampling: uniform (random per time step), constant (fixed spatial points), shifty (well-refined varying with time)')
    p.add_argument('--n_coll_points', type=int, default=30000,
                   help='Number of collocation points (per time step)')
    p.add_argument('--use_obse_4x4', type=str2bool, default=True,
                   help='Whether to use 4x4 observation points (true/false)')

    # Temporal Causality Strategy
    # Core idea: Use only first few time steps early in training, gradually add more as training progresses
    p.add_argument('--use_time_causal', type=str2bool, default=True,
                   help='Whether to enable temporal causality strategy (true/false)')
    p.add_argument('--time_causal_coeff', type=int, default=100,
                   help='Temporal causality coefficient: epochs between adding each time step')
    p.add_argument('--time_causal_steps', type=int, default=200,
                   help='Total time steps for temporal causality strategy (use all data after reaching this)')

    # Paths
    p.add_argument('--data_path', type=str, default='../pre_pkl_data/')
    p.add_argument('--save_path', type=str, default='./results/')

    # Load model
    p.add_argument('--eval_only', action='store_true', default=False,
                   help='Evaluation only mode, no training')

    args = p.parse_args()

    # Auto-calculate batch sizes
    args.batch_size_ic = 10000
    args.batch_size_diri = 10000
    args.batch_size_neum = 10000  # Neumann BC batch size
    args.time_causal_coeff = args.epochs//5//args.time_causal_steps

    return args
