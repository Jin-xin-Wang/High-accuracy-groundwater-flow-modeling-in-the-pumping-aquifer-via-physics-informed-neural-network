# -*- coding: utf-8 -*-
"""
Main Program - Heterogeneous Single-Well Pumping Flow PINN (Fixed Head Boundary)
"""
import os
import time
import torch
import numpy as np

from config import get_args, TrainingConfig, DomainConfig
from utils import (
    device, set_seed, make_dir, write_pkl,
    setup_logger, count_parameters, save_checkpoint, load_checkpoint, mse_loss,
    GPUDataLoader, plot_comparison, plot_loss_history
)
from models import get_model, PINN_Physics
from data_processor import DataProcessor

from datetime import datetime


class Trainer:
    """Trainer class for PINN training"""

    def __init__(self, args, data_dict, scale_cfg, logger):
        self.args = args
        self.data = data_dict
        self.scale_cfg = scale_cfg
        self.logger = logger

        # Physics constraints
        self.physics = PINN_Physics(scale_cfg)

        # Initialize model
        self.model = get_model(args.model_name, args).to(device)
        self.logger.info(f"Using model: {args.model_name}")
        count_parameters(self.model, log=self.logger)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # Learning rate scheduler (ensure milestone interval is at least 1)
        step_size = max(1, args.epochs // 30)
        milestones = [i * step_size for i in range(1, 100) if i * step_size < args.epochs]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.9
        )

        # Loss history
        self.loss_history = {
            'epoch': [],
            'ic': [],
            'diri': [],
            'pde': [],
            'obse': [],
            'well': [],
        }

        # ============ Temporal Causality Strategy Parameters ============
        self.use_time_causal = args.use_time_causal
        self.time_causal_coeff = args.time_causal_coeff
        self.time_causal_steps = args.time_causal_steps

        if self.use_time_causal:
            # Time step info for collocation data
            self.n_time_steps_coll = data_dict.get('n_time_steps_coll', args.n_time_steps)
            # Time step info for well constraint data
            self.n_time_steps_well = data_dict.get('n_time_steps_well', 100)
            # Current number of time steps used (starting from 1)
            self.current_time_idx = 1
            # Epoch counter (for determining when to increase time steps)
            self.causal_epoch_counter = 0
            # Epoch when temporal causality strategy ends
            self.causal_end_epoch = self.time_causal_coeff * self.time_causal_steps

            self.logger.info(f"Temporal causality strategy enabled:")
            self.logger.info(f"  - Collocation data time steps: {self.n_time_steps_coll}")
            self.logger.info(f"  - Well constraint time steps: {self.n_time_steps_well}")
            self.logger.info(f"  - Increase one time step every {self.time_causal_coeff} epochs")
            self.logger.info(f"  - Causality strategy lasts {self.time_causal_steps} time steps")
            self.logger.info(f"  - Causality strategy ends at epoch: {self.causal_end_epoch}")

        # Setup data loaders
        self._setup_dataloaders()

    def _setup_dataloaders(self):
        """Setup data loaders"""
        args = self.args

        # ============ Temporal Causality Strategy: Data organized by time steps ============
        if self.use_time_causal:
            # Collocation data organized by time steps: (n_time_steps, n_points_per_step, features)
            self.coll_by_time = self.data['coll_inputs_by_time']
            # Well constraint data organized by time steps
            self.well_by_time_inputs = self.data['well_inputs_by_time']
            self.well_by_time_labels = self.data['well_labels_by_time']
            self.well_by_time_K = self.data['well_K_by_time']

            # Initialize collocation data loader (only use first time step)
            self._update_coll_loader(self.current_time_idx)
            # Initialize well constraint data (only use first time step)
            self._update_well_data(self.current_time_idx)
        else:
            # Original logic: Collocation data (GPU DataLoader)
            self.coll_loader = GPUDataLoader(
                self.data['coll_inputs'],
                batch_size=args.batch_size_coll
            )
            self.coll_iter = iter(self.coll_loader)

            # Well constraints (all data)
            self.well_inputs = self.data['well_inputs'].requires_grad_(True)
            self.well_labels = self.data['well_labels']
            self.well_K = self.data['well_K']

        # Initial condition (GPU DataLoader) - not affected by temporal causality strategy
        self.ic_loader = GPUDataLoader(
            self.data['ic_inputs'].requires_grad_(False),
            self.data['ic_labels'],
            batch_size=args.batch_size_ic
        )
        self.ic_iter = iter(self.ic_loader)

        # Dirichlet boundary condition - not affected by temporal causality strategy
        self.diri_loader = GPUDataLoader(
            self.data['diri_inputs'].requires_grad_(False),
            self.data['diri_labels'],
            batch_size=args.batch_size_diri
        )
        self.diri_iter = iter(self.diri_loader)

        # Observation data - not affected by temporal causality strategy
        self.obse_inputs = self.data['obse_inputs'].requires_grad_(False)
        self.obse_labels = self.data['obse_labels']

        # Test data
        self.test_txy = self.data['test_txy'].requires_grad_(False)
        self.test_head = self.data['test_head']
        self.plot_x = self.data['plot_x']
        self.plot_y = self.data['plot_y']

    def _update_coll_loader(self, n_steps):
        """
        Update collocation data loader, only use data from first n_steps time steps

        Core method for temporal causality strategy:
        - Take collocation data organized by time steps (n_time_steps, n_points_per_step, features)
        - Select first n_steps time steps
        - Flatten to (n_steps * n_points_per_step, features) for training

        Args:
            n_steps: Number of time steps to use
        """
        # Select data from first n_steps time steps
        coll_data = self.coll_by_time[:n_steps]  # (n_steps, n_points_per_step, features)
        # Flatten to 2D tensor
        coll_data = coll_data.reshape(-1, coll_data.shape[-1])  # (n_steps * n_points_per_step, features)

        # Create new data loader
        self.coll_loader = GPUDataLoader(coll_data, batch_size=self.args.batch_size_coll)
        self.coll_iter = iter(self.coll_loader)

    def _update_well_data(self, n_steps):
        """
        Update well constraint data, only use data from first n_steps time steps

        Core method for temporal causality strategy:
        - Select first n_steps time steps from well constraint data organized by time
        - Flatten for training

        Args:
            n_steps: Number of time steps to use
        """
        # Calculate corresponding time steps for well constraints (may differ from collocation)
        # Calculate proportionally: well_steps = n_steps * (n_time_steps_well / n_time_steps_coll)
        well_steps = max(1, int(n_steps * self.n_time_steps_well / self.n_time_steps_coll))
        well_steps = min(well_steps, self.n_time_steps_well)  # Don't exceed total time steps

        # Select data from first well_steps time steps and flatten
        well_inputs = self.well_by_time_inputs[:well_steps]
        well_labels = self.well_by_time_labels[:well_steps]
        well_K = self.well_by_time_K[:well_steps]

        self.well_inputs = well_inputs.reshape(-1, well_inputs.shape[-1]).requires_grad_(True)
        self.well_labels = well_labels.reshape(-1, well_labels.shape[-1])
        self.well_K = well_K.reshape(-1, well_K.shape[-1])

    def _update_time_causal(self, epoch):
        """
        Temporal causality strategy: Update training data time range based on current epoch

        Core logic:
        1. During causal training phase (epoch < causal_end_epoch):
           - Increase one time step every time_causal_coeff epochs
           - Update collocation data loader and well constraint data
        2. When causal training phase ends:
           - Switch to using all time steps data

        Args:
            epoch: Current epoch number
        """
        # Check if in causal training phase
        if epoch < self.causal_end_epoch:
            # Increment epoch counter
            self.causal_epoch_counter += 1

            # Increase one time step every time_causal_coeff epochs
            if self.causal_epoch_counter >= self.time_causal_coeff:
                self.causal_epoch_counter = 0  # Reset counter
                self.current_time_idx += 1     # Increase time step

                # Ensure not exceeding total time steps
                self.current_time_idx = min(self.current_time_idx, self.n_time_steps_coll)

                # Update collocation data loader
                self._update_coll_loader(self.current_time_idx)
                # Update well constraint data
                self._update_well_data(self.current_time_idx)

                # Log
                self.logger.info(f"[Temporal Causal] Epoch {epoch+1}: Expanded to {self.current_time_idx}/{self.n_time_steps_coll} time steps")

        elif epoch == self.causal_end_epoch:
            # Causal training phase ends, switch to all data
            self.current_time_idx = self.n_time_steps_coll
            self._update_coll_loader(self.current_time_idx)
            self._update_well_data(self.current_time_idx)
            self.logger.info(f"[Temporal Causal] Epoch {epoch+1}: Causality strategy ended, using all {self.n_time_steps_coll} time steps")

    def train(self):
        """Main training loop"""
        args = self.args
        epochs = args.epochs

        # Log interval
        log_interval = max(1, epochs // 1000)
        save_interval = max(1, epochs // 50)

        self.logger.info(f"Starting training, total epochs: {epochs}")
        self.logger.info(f"batch_size: coll={args.batch_size_coll}, ic={args.batch_size_ic}, diri={args.batch_size_diri}")
        self.logger.info(f"Time sampling: {args.time_sampling}, use 4x4 observation points: {args.use_obse_4x4}")
        if self.use_time_causal:
            self.logger.info(f"Temporal causality strategy: enabled (coeff={self.time_causal_coeff}, steps={self.time_causal_steps})")
        t0 = time.time()

        self.model.train()

        for epoch in range(epochs):
            # ============ Temporal Causality Strategy: Gradually increase time steps ============
            if self.use_time_causal:
                self._update_time_causal(epoch)

            # Dynamic adjustment of well constraint weight
            # w_well = 100 * (0.5 ** (epoch / (epochs // 20))) + 4
            w_well = 100 * (0.5 ** (epoch / 10000)) + 1

            self.optimizer.zero_grad()

            # ============ Calculate various losses ============

            # 1. Collocation PDE loss
            coll_batch = next(self.coll_iter).requires_grad_(True)
            coll_pred = self.model.forward(coll_batch[..., :3])  # Only input t,x,y
            pde_loss = self.physics.get_pde_residual(coll_batch, coll_pred)

            # 2. Initial condition loss
            ic_inputs, ic_labels = next(self.ic_iter)
            ic_pred = self.model.forward(ic_inputs)
            ic_loss = mse_loss(ic_pred, ic_labels)

            # 3. Dirichlet boundary condition loss
            diri_inputs, diri_labels = next(self.diri_iter)
            diri_pred = self.model.forward(diri_inputs)
            diri_loss = mse_loss(diri_pred, diri_labels)

            # 4. Observation data loss
            obse_pred = self.model.forward(self.obse_inputs)
            obse_loss = mse_loss(obse_pred, self.obse_labels)

            # 5. Well flux constraint loss
            well_pred = self.model.forward(self.well_inputs)
            well_loss = self.physics.get_well_flux_loss(
                self.well_inputs,
                well_pred,
                self.well_labels,
                self.well_K
            )

            # ============ Total loss ============
            loss = (5.0 * ic_loss +
                    1.0 * diri_loss +
                    1.0 * pde_loss +
                    2.0 * obse_loss +
                    w_well * well_loss)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # ============ Logging ============
            if (epoch + 1) % log_interval == 0:
                self.loss_history['epoch'].append(epoch + 1)
                self.loss_history['ic'].append(ic_loss.item())
                self.loss_history['diri'].append(diri_loss.item())
                self.loss_history['pde'].append(pde_loss.item())
                self.loss_history['obse'].append(obse_loss.item())
                self.loss_history['well'].append(well_loss.item())

                elapsed = (time.time() - t0) / 60
                self.logger.info(
                    f'[{epoch + 1}/{epochs} {(epoch + 1) / epochs * 100:.1f}%] '
                    f'ic:{ic_loss.item():.3e} diri:{diri_loss.item():.3e} '
                    f'pde:{pde_loss.item():.3e} obse:{obse_loss.item():.3e} well:{well_loss.item():.3e} '
                    f'time:{elapsed:.1f}min'
                )

            # ============ Save checkpoint ============
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch + 1)
                self._evaluate_and_plot(epoch + 1)

        # Save at end of training
        self._save_checkpoint(epochs)
        self._evaluate_and_plot(epochs)

        self.logger.info(f"Training completed! Total time: {(time.time() - t0) / 60:.1f} min")

        return self.model

    def _save_checkpoint(self, epoch):
        """Save checkpoint (including model parameters, optimizer state, loss history)"""
        save_checkpoint(
            self.model, self.optimizer, epoch,
            os.path.join(self.args.save_path, f'checkpoint_{epoch}.pt'),
            loss_history=self.loss_history
        )
        # Also save separate loss_history.pkl for easy viewing
        write_pkl(self.loss_history, os.path.join(self.args.save_path, 'loss_history.pkl'))

    def _evaluate_and_plot(self, epoch, save_eps=False):
        """Evaluate and plot"""
        self.model.eval()

        with torch.no_grad():
            # Predict test data (dynamically get number of snapshots)
            n_snapshots = self.test_txy.shape[0]
            pred_list = []
            for i in range(n_snapshots):
                pred = self.model(self.test_txy[i]).cpu().numpy()[:, 0]
                pred_list.append(pred)
            pred_head = np.stack(pred_list, axis=0)

            # Denormalize
            pred_head_real = self.scale_cfg.denormalize_h(pred_head)
            ref_head_real = self.scale_cfg.denormalize_h(self.test_head.cpu().numpy())

            # Calculate errors
            mae = np.abs(ref_head_real - pred_head_real).mean()
            rel_l2 = np.linalg.norm(ref_head_real.flatten() - pred_head_real.flatten()) / \
                     np.linalg.norm(ref_head_real.flatten())

            self.logger.info(f'Epoch {epoch}: MAE={mae:.4f}m, rL2={rel_l2:.4f}')

        # Plot comparison (use actual snapshot times)
        times = self.data.get('test_times', [0.05, 0.5, 5.0])[:n_snapshots]
        plot_comparison(
            self.plot_x.cpu().numpy(), self.plot_y.cpu().numpy(),
            ref_head_real, pred_head_real, times,
            save_path=os.path.join(self.args.save_path, f'comparison_{epoch}'),
            save_eps=save_eps
        )

        # Plot loss curves
        plot_loss_history(
            self.loss_history['epoch'],
            {k: v for k, v in self.loss_history.items() if k != 'epoch'},
            save_path=os.path.join(self.args.save_path, f'loss_{epoch}'),
            save_eps=save_eps
        )

        self.model.train()


def main():
    """Main function"""
    # Get arguments
    args = get_args()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d-%H-%M")

    # Set save path (including temporal causality strategy identifier)
    causal_str = f"_causal{args.time_causal_coeff}x{args.time_causal_steps}" if args.use_time_causal else ""
    args.save_path = f'./results/{args.model_name}_epochs{args.epochs}_bs{args.batch_size_coll}_{args.time_sampling}_nonlinear-{args.t_nonlinear}{causal_str}-{date_str}/'
    make_dir(args.save_path)

    # Set random seed
    set_seed(args.seed)

    # Setup logger
    logger = setup_logger(os.path.join(args.save_path, 'train.log'))


    # ============ Data Processing ============
    logger.info("=" * 50)
    logger.info("Data Processing")
    logger.info("=" * 50)

    # Create shared domain config (single source of truth for t_min/t_max)
    domain_cfg = DomainConfig()

    # Create training config (initialized from args, linked to domain_cfg)
    train_cfg = TrainingConfig(args, domain_cfg=domain_cfg)

    # Data processing
    data_path = args.data_path
    processor = DataProcessor(data_path, train_cfg=train_cfg, logger=logger)
    scale_cfg = processor.scale_cfg

    # Prepare data
    data_dict = processor.prepare_all_data(
        save_path=os.path.join(args.save_path, 'processed_data.pkl')
    )

    # Convert to torch tensor
    data_dict = processor.to_torch(data_dict)

    # ============ Create Trainer ============
    trainer = Trainer(args, data_dict, scale_cfg, logger)


    logger.info(f"Configuration: {args}")

    # ============ Load Existing Model ============
    if False:
        args.load_checkpoint = 'results/PirateNet_T1S2_epochs300000_bs3276_ln/checkpoint_300000.pt'
        args.eval_only = True


        logger.info("=" * 50)
        logger.info(f"Loading model: {args.load_checkpoint}")
        logger.info("=" * 50)

        trainer.model, trainer.optimizer, loaded_epoch, loaded_loss_history = load_checkpoint(
            trainer.model, trainer.optimizer, args.load_checkpoint, log=logger
        )
        # Restore loss history (for continued training or plotting)
        if loaded_loss_history is not None:
            trainer.loss_history = loaded_loss_history
        logger.info(f"Successfully loaded checkpoint, trained for {loaded_epoch} epochs")

        # Evaluation only mode
        if args.eval_only:
            logger.info("=" * 50)
            logger.info("Evaluation Only Mode")
            logger.info("=" * 50)
            trainer._evaluate_and_plot(loaded_epoch, save_eps=True)
            logger.info("Evaluation completed")
            return

    # ============ Training ============
    logger.info("=" * 50)
    logger.info("Model Training")
    logger.info("=" * 50)

    model = trainer.train()

    logger.info("Program finished")


if __name__ == "__main__":
    main()
