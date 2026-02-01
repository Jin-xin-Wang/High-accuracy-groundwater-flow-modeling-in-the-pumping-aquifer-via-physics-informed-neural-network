# -*- coding: utf-8 -*-
"""
Utility Functions - Groundwater Flow PINN
"""
import os
import random
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Configure module-level logger
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

plt.rcParams['figure.max_open_warning'] = 4
plt.rcParams['image.cmap'] = 'jet'


# ============== Device Configuration ==============
if torch.cuda.is_available():
    logger.info('CUDA available')
    device = torch.device('cuda')
else:
    logger.info('CUDA not available, using CPU')
    device = torch.device('cpu')


# ============== File Operations ==============
def write_pkl(data, pkl_path):
    """Save pickle file"""
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def read_pkl(pkl_path):
    """Read pickle file"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def make_dir(dir_path):
    """Create directory"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# ============== Random Seed ==============
def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_dtype(torch.float32)


# ============== Loss Functions ==============
def mse_loss(pred, target):
    """MSE loss"""
    return torch.nn.MSELoss()(pred, target)


# ============== Model Utilities ==============
def count_parameters(model, log=None):
    """Count trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = f'Total trainable parameters: {total:,}'
    if log:
        log.info(msg)
    else:
        logger.info(msg)
    return total


def save_checkpoint(model, optimizer, epoch, path, loss_history=None):
    """
    Save checkpoint

    Args:
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        path: Save path
        loss_history: Loss history dict (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if loss_history is not None:
        checkpoint['loss_history'] = loss_history
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, log=None):
    """
    Load checkpoint

    Returns:
        model, optimizer, epoch, loss_history (if exists)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss_history = checkpoint.get('loss_history', None)
    msg = f'Model loaded! Epoch: {epoch}'
    if log:
        log.info(msg)
    else:
        logger.info(msg)
    return model, optimizer, epoch, loss_history


# ============== Activation Functions ==============
class Sin(nn.Module):
    """Sin activation function"""
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


def get_activation(name):
    """Get activation function by name"""
    activations = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'sin': Sin,
        'silu': nn.SiLU,
    }
    return activations.get(name, nn.Tanh)


# ============== Network Initialization ==============
def init_weights(module, method='kaiming'):
    """Initialize network weights"""
    if method == 'xavier':
        init_fn = nn.init.xavier_uniform_
    elif method == 'kaiming':
        init_fn = nn.init.kaiming_normal_
    else:
        init_fn = nn.init.xavier_uniform_

    if isinstance(module, nn.Linear):
        init_fn(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ============== Data Loader ==============
class GPUDataLoader:
    """GPU Data Loader - Infinite random sampling version, avoids CPU-GPU data transfer"""
    def __init__(self, *tensors, batch_size=32):
        self.tensors = tensors
        self.batch_size = batch_size
        self.n_samples = tensors[0].shape[0]
        self.device = tensors[0].device

    def __iter__(self):
        while True:
            # Random sample batch_size indices each time (with replacement)
            indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
            if len(self.tensors) == 1:
                yield self.tensors[0][indices]
            else:
                yield tuple(t[indices] for t in self.tensors)


# ============== Logging ==============
def setup_logger(log_path, name='PINN'):
    """Setup logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============== Visualization ==============
def plot_field(x, y, data, title='', save_path=None, vmin=None, vmax=None):
    """Plot field distribution"""
    fig, ax = plt.subplots(figsize=(8, 6))
    # Specify vmin/vmax when plotting
    im = ax.tricontourf(x, y, data, levels=100, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_comparison(x, y, ref_data, pred_data, times, save_path=None, save_eps=False):
    """Plot reference vs prediction comparison"""
    n_times = len(times)
    fig, axes = plt.subplots(n_times, 3, figsize=(15, 4*n_times))

    if n_times == 1:
        axes = axes.reshape(1, -1)

    titles = ['Reference', 'Prediction', 'Error']

    for i, t in enumerate(times):
        ref = ref_data[i]
        pred = pred_data[i]
        error = ref - pred

        data_list = [ref, pred, error]

        for j in range(3):
            ax = axes[i, j]
            im = ax.tricontourf(x, y, data_list[j], levels=100)
            fig.colorbar(im, ax=ax)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=14)
            if j == 0:
                ax.set_ylabel(f't={t}d', fontsize=14)

    # Calculate error metrics
    mae = np.abs(ref_data - pred_data).mean()
    rel_l2 = np.linalg.norm(ref_data.flatten() - pred_data.flatten()) / np.linalg.norm(ref_data.flatten())

    plt.suptitle(f'MAE={mae:.4f}, rL2={rel_l2:.4f}', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        if save_eps:
            plt.savefig(save_path+'.eps', dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_loss_history(epoch_list, loss_dict, save_path=None, save_eps=False):
    """Plot loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, losses in loss_dict.items():
        ax.plot(epoch_list, losses, label=name)

    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+'.png', dpi=150)
        if save_eps:
            plt.savefig(save_path+'.eps', dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig


# ============== Automatic Differentiation Utilities ==============
def grad(outputs, inputs):
    """Compute gradients"""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=True,
        create_graph=True
    )[0]
