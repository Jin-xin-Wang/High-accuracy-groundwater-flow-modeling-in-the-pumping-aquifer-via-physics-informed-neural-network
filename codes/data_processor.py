# -*- coding: utf-8 -*-
"""
Data Processing Module - Heterogeneous Single-Well Pumping Flow PINN (Fixed Head Boundary on All Sides)

Functions:
1. Load raw data exported from COMSOL
2. Generate various training data (IC, BC, collocation points, observation points, well constraints)
3. Unified dimensionless processing
"""
import logging
import os
import glob
import numpy as np
import torch
from scipy.spatial import Delaunay
from config import ScaleConfig, WellConfig, DomainConfig, TrainingConfig
from utils import read_pkl, write_pkl, device

# Module-level logger
module_logger = logging.getLogger(__name__)


# ============== Spatial Sampling Helper Functions ==============
def is_in_poly(points, poly):
    """
    Check if points are inside a polygon

    Args:
        points: (n, 2) array of spatial point coordinates
        poly: list of polygon vertices [[x1,y1], [x2,y2], ...]

    Returns:
        Boolean array, True indicates inside the polygon
    """
    is_in_list = []
    for idx in range(len(points)):
        is_in = False
        px, py = points[idx, 0], points[idx, 1]
        for i, corner in enumerate(poly):
            next_i = i + 1 if i + 1 < len(poly) else 0
            x1, y1 = corner
            x2, y2 = poly[next_i]
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):
                is_in = True
                break
            if min(y1, y2) < py <= max(y1, y2):
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if x == px:
                    is_in = True
                    break
                elif x > px:
                    is_in = not is_in
        is_in_list.append(is_in)
    return np.array(is_in_list)


def is_retain_not_close_wells(points, wells_coord_list, discard_radius):
    """
    Check if points are far from well locations

    Args:
        points: (n, 2) array of spatial point coordinates
        wells_coord_list: list of well coordinates [[x1,y1], [x2,y2], ...]
        discard_radius: discard radius, points closer than this value to wells are discarded

    Returns:
        Boolean array, True indicates retained (far from wells)
    """
    is_retain_list = []
    is_retain_criteria = discard_radius ** 2

    for well_coord in wells_coord_list:
        is_retain_list.append(np.sum(np.square(points - well_coord), axis=-1) >= is_retain_criteria)

    return np.prod(is_retain_list, axis=0).astype(bool)


def LR_sampling(center_list, radius, factor, all_num_points, num_factor=100):
    """
    Local Refinement Sampling around wells

    Performs exponential decay probability sampling around each center point,
    with higher sampling probability closer to the center.

    Sampling probability formula:
        p(r) = exp(-r/R * f) / Σexp(-r/R * f)

    Where:
        r: distance from point to center
        R: sampling radius
        f: refinement factor, larger values result in denser sampling near center

    Args:
        center_list: list of sampling center points [[x1,y1], [x2,y2], ...]
        radius: sampling radius
        factor: refinement factor, larger values increase sampling probability near center
        all_num_points: total number of sampling points
        num_factor: multiplier for generating candidate points

    Returns:
        points: (n, 2) array of sampled point coordinates
    """
    points_list = []
    num_points = int(all_num_points / len(center_list))

    for center in center_list:
        # Generate candidate points uniformly in [-2R, 2R] × [-2R, 2R] range
        points = np.random.uniform(-radius * 2, radius * 2, size=(num_points * num_factor, 2))

        # Keep only points inside the circle
        points = points[np.linalg.norm(points, axis=1) <= radius]

        # Calculate distance from each point to center
        distances = np.linalg.norm(points, axis=1)

        # Calculate sampling probability: p(r) ∝ exp(-r/R * factor)
        probabilities = np.exp(-distances / radius * factor)
        probabilities /= np.sum(probabilities)

        # Sample according to probability
        sampled_indices = np.random.choice(len(points), size=min(num_points, len(points)),
                                           p=probabilities, replace=False)
        points_list.append(points[sampled_indices] + center)

    points = np.vstack(points_list)
    np.random.shuffle(points)
    return points


class TriangularInterpolator:
    """
    Triangular mesh-based hydraulic conductivity interpolator

    Uses Delaunay triangulation and linear shape functions for interpolation.
    Can simultaneously compute K values and their spatial derivatives Kx, Ky.

    Shape function formulas (barycentric coordinates):
        det = (y2-y3)(x1-x3) + (x3-x2)(y1-y3)
        N1 = [(y2-y3)(x-x3) + (x3-x2)(y-y3)] / det
        N2 = [(y3-y1)(x-x3) + (x1-x3)(y-y3)] / det
        N3 = 1 - N1 - N2

    Interpolation formula:
        K(x,y) = N1*K1 + N2*K2 + N3*K3

    Derivative formulas:
        Kx = dK/dx = (y2-y3)/det * K1 + (y3-y1)/det * K2 + (y1-y2)/det * K3
        Ky = dK/dy = (x3-x2)/det * K1 + (x1-x3)/det * K2 + (x2-x1)/det * K3
    """

    def __init__(self, xyK):
        """
        Initialize triangular mesh interpolator

        Args:
            xyK: (n, 3) array, columns are [x, y, K]
        """
        self.points = xyK[:, :2]  # (n, 2) spatial coordinates
        self.K_values = xyK[:, 2]  # (n,) K values

        # Build Delaunay triangular mesh
        self.tri = Delaunay(self.points)

        # Precompute shape function derivatives for each triangle
        self._precompute_gradients()

    def _precompute_gradients(self):
        """Precompute shape function derivatives for each triangle"""
        n_triangles = len(self.tri.simplices)
        self.dN_dx = np.zeros((n_triangles, 3))
        self.dN_dy = np.zeros((n_triangles, 3))
        self.det = np.zeros(n_triangles)

        for i, simplex in enumerate(self.tri.simplices):
            x1, y1 = self.points[simplex[0]]
            x2, y2 = self.points[simplex[1]]
            x3, y3 = self.points[simplex[2]]

            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            self.det[i] = det

            if abs(det) < 1e-12:
                continue

            self.dN_dx[i, 0] = (y2 - y3) / det
            self.dN_dx[i, 1] = (y3 - y1) / det
            self.dN_dx[i, 2] = (y1 - y2) / det

            self.dN_dy[i, 0] = (x3 - x2) / det
            self.dN_dy[i, 1] = (x1 - x3) / det
            self.dN_dy[i, 2] = (x2 - x1) / det

    def interpolate(self, xy, return_gradient=True):
        """
        Interpolate at given points

        Args:
            xy: (n, 2) query point coordinates
            return_gradient: whether to return derivatives Kx, Ky

        Returns:
            K, Kx, Ky or only K
        """
        n_points = len(xy)
        K = np.zeros(n_points)
        Kx = np.zeros(n_points)
        Ky = np.zeros(n_points)

        tri_indices = self.tri.find_simplex(xy)

        for i in range(n_points):
            tri_idx = tri_indices[i]

            if tri_idx == -1:
                # Point is outside the mesh, use nearest neighbor
                dists = np.sqrt(np.sum((self.points - xy[i]) ** 2, axis=1))
                nearest_idx = np.argmin(dists)
                K[i] = self.K_values[nearest_idx]
                Kx[i] = 0.0
                Ky[i] = 0.0
            else:
                simplex = self.tri.simplices[tri_idx]
                x1, y1 = self.points[simplex[0]]
                x2, y2 = self.points[simplex[1]]
                x3, y3 = self.points[simplex[2]]

                K1 = self.K_values[simplex[0]]
                K2 = self.K_values[simplex[1]]
                K3 = self.K_values[simplex[2]]

                x, y = xy[i]
                det = self.det[tri_idx]

                if abs(det) < 1e-12:
                    K[i] = (K1 + K2 + K3) / 3
                    Kx[i] = 0.0
                    Ky[i] = 0.0
                else:
                    N1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
                    N2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
                    N3 = 1 - N1 - N2

                    K[i] = N1 * K1 + N2 * K2 + N3 * K3

                    if return_gradient:
                        Kx[i] = (self.dN_dx[tri_idx, 0] * K1 +
                                self.dN_dx[tri_idx, 1] * K2 +
                                self.dN_dx[tri_idx, 2] * K3)
                        Ky[i] = (self.dN_dy[tri_idx, 0] * K1 +
                                self.dN_dy[tri_idx, 1] * K2 +
                                self.dN_dy[tri_idx, 2] * K3)

        if return_gradient:
            return K, Kx, Ky
        else:
            return K


class DataProcessor:
    """Data Processor"""

    def __init__(self, data_path, train_cfg=None, scale_config=None, logger=None):
        """
        Initialize data processor

        Args:
            data_path: raw data path (pre_pkl_data/)
            train_cfg: training configuration
            scale_config: scaling configuration, auto-computed if None
            logger: logger (optional, defaults to module logger)
        """
        self.logger = logger if logger else module_logger
        self.data_path = data_path
        self.domain_cfg = DomainConfig()
        self.well_cfg = WellConfig()
        # Ensure train_cfg is linked with domain_cfg: t_min/t_max from domain_cfg
        if train_cfg is None:
            self.train_cfg = TrainingConfig(domain_cfg=self.domain_cfg)
        else:
            # External train_cfg, update its domain_cfg to current instance
            train_cfg.domain_cfg = self.domain_cfg
            self.train_cfg = train_cfg

        # Load raw data
        self._load_raw_data()

        # Initialize scaling configuration
        if scale_config is None:
            h_max = np.abs(self.well_head).max()
            self.logger.info(f"Maximum absolute well head value: {h_max:.3f} m")
            self.scale_cfg = ScaleConfig(h_max=h_max)
        else:
            self.scale_cfg = scale_config

        self.logger.info(f"Dimensionless parameters:")
        self.logger.info(f"  x_scale = {self.scale_cfg.x_scale} m")
        self.logger.info(f"  t_scale = {self.scale_cfg.t_scale} d")
        self.logger.info(f"  h_scale = {self.scale_cfg.h_scale:.3f} m")

    def _load_raw_data(self):
        # Robust loader for:
        # - single-well datasets exported as 1D arrays
        # - Windows filename encoding differences (e.g. '×' vs garbled char)
        #
        # We return early to avoid the legacy assumptions below.
        self.logger.info("Loading raw data (robust mode)...")

        well_data = np.asarray(read_pkl(self.data_path + 'well_data.pkl'))
        if well_data.ndim == 1:
            well_data = well_data.reshape(1, -1)
        if well_data.ndim != 2 or well_data.shape[1] < 3:
            raise ValueError(f"Unexpected well_data shape: {well_data.shape}")
        self.well_pos = well_data[:, :2]
        self.well_head = well_data[:, 2:]

        # Sync WellConfig from data so that sampling/constraints match the dataset.
        self.well_cfg.positions = self.well_pos.copy()
        self.well_cfg.n_wells = int(self.well_pos.shape[0])
        if (not hasattr(self.well_cfg, 'well_types')) or (len(self.well_cfg.well_types) != self.well_cfg.n_wells):
            self.well_cfg.well_types = np.full(self.well_cfg.n_wells, -1, dtype=int)

        well_K = np.asarray(read_pkl(self.data_path + 'well_K.pkl'))
        if well_K.ndim == 1:
            well_K = well_K.reshape(self.well_cfg.n_wells, -1)
        self.well_K = well_K

        # Observation points: tolerate filename encoding differences on Windows.
        try:
            obse_data = read_pkl(self.data_path + 'obse_data_4×4.pkl')
        except FileNotFoundError:
            candidates = sorted(glob.glob(os.path.join(self.data_path, 'obse_data_4*4.pkl')))
            if not candidates:
                raise
            obse_data = read_pkl(candidates[0])
        self.obse_pos = obse_data[:, :2]
        self.obse_head = obse_data[:, 2:]

        # Snapshots for test/plotting.
        self.snap_txy = read_pkl(self.data_path + 'snap_txy.pkl')
        self.snap_data = read_pkl(self.data_path + 'snap_data.pkl')

        # K-field for PDE residuals.
        self.xyK = read_pkl(self.data_path + 'xyK.pkl')
        self.K_interpolator = TriangularInterpolator(self.xyK)

        # Time meta
        self.n_time_steps_data = self.well_head.shape[1]
        self.snap_times = [self.snap_txy[i, 0, 0] for i in range(self.snap_txy.shape[0])]

        self.logger.info(f"  n_wells: {self.well_cfg.n_wells}")
        self.logger.info(f"  n_time_steps_data: {self.n_time_steps_data}")
        self.logger.info(f"  snap_times: {self.snap_times}")
        return
        """Load raw data"""
        self.logger.info("Loading raw data...")

        # Well data: (5, 203) - [x, y, h0, h1, ..., h200]
        well_data = read_pkl(self.data_path + 'well_data.pkl')
        self.well_pos = well_data[:, :2]      # (5, 2) well positions
        self.well_head = well_data[:, 2:]     # (5, 201) well head time series

        # K values for well flow constraints: (5, 201) - K values for 5 wells at 201 time steps
        self.well_K = read_pkl(self.data_path + 'well_K.pkl')

        # Observation point data: (16, 203) - [x, y, h0, h1, ..., h200]
        obse_data = read_pkl(self.data_path + 'obse_data_4×4.pkl')
        self.obse_pos = obse_data[:, :2]      # (16, 2) observation point positions
        self.obse_head = obse_data[:, 2:]     # (16, 201) observation point head time series

        # Snapshot spatiotemporal coordinates: (4, 16209, 3) - [t, x, y]
        self.snap_txy = read_pkl(self.data_path + 'snap_txy.pkl')

        # Snapshot data: (4, 4, 16209) - [H, ?, u, v]
        self.snap_data = read_pkl(self.data_path + 'snap_data.pkl')

        # Hydraulic conductivity field: (n, 3) - [x, y, K]
        # Use triangular mesh interpolation to compute K, Kx, Ky
        self.xyK = read_pkl(self.data_path + 'xyK.pkl')
        self.K_interpolator = TriangularInterpolator(self.xyK)
        self.logger.info(f"  Hydraulic conductivity field points: {self.xyK.shape[0]}")
        self.logger.info(f"  Triangular mesh elements: {len(self.K_interpolator.tri.simplices)}")

        # Get time step information
        self.n_time_steps_data = self.well_head.shape[1]
        self.snap_times = [self.snap_txy[i, 0, 0] for i in range(4)]

        self.logger.info(f"  Well/observation time steps: {self.n_time_steps_data}")
        self.logger.info(f"  Snapshot times: {self.snap_times}")

    def generate_time_array(self, n=None):
        """
        Generate time sampling array

        Select sampling method based on train_cfg.time_sampling:
        - 'uniform': uniform sampling
        - 'lg': logarithmic sampling (log10)
        - 'ln': natural logarithmic sampling

        Returns:
            t_array: (n_time_steps,) time array [t_min, t_max]
        """
        cfg = self.train_cfg
        if n is None:
            n = cfg.n_time_steps
        t_min = cfg.t_min
        t_max = cfg.t_max*1.02

        if cfg.time_sampling == 'uniform':
            t_array = np.linspace(t_min, t_max, n)
        elif cfg.time_sampling == 'lg':
            # log10 sampling: uniform distribution in log10 space
            t_array = np.logspace(np.log10(t_min), np.log10(t_max), n)
        else:
            raise ValueError(f"Unknown time_sampling: {cfg.time_sampling}")

        return t_array


    def generate_ic_data(self):
        """
        Generate initial condition data (raw data, not dimensionless)

        Uses spatial coordinates from snapshot data, t=0, h=0.002*(500-x)
        Initial state has a linear background flow field from left to right

        Returns:
            ic_data: dict with keys 't', 'x', 'y', 'h'
        """
        self.logger.info("Generating initial condition data...")

        # Use spatial points from snapshot data (snap_txy[0] first snapshot coordinates)
        x = self.snap_txy[0, :, 1].copy()  # snap_txy: [t, x, y]
        y = self.snap_txy[0, :, 2].copy()

        n_points = len(x)

        t = np.zeros(n_points)
        # Case update: initial condition is uniform 0 m everywhere.
        h = np.zeros(n_points)

        ic_data = {
            't': t,
            'x': x,
            'y': y,
            'h': h
        }

        self.logger.info(f"  Initial condition points: {n_points}")
        self.logger.info(f"  Initial condition: h = 0m (uniform)")
        return ic_data

    def generate_bc_dirichlet_data(self):
        """
        Generate Dirichlet boundary condition data (raw data, not dimensionless)

        All four boundaries are fixed head boundaries h=0m

        Returns:
            bc_data: dict with keys 't', 'x', 'y', 'h'
        """
        self.logger.info("Generating Dirichlet boundary condition data (fixed head h=0m on all sides)...")

        t_array = self.generate_time_array()
        n_points_per_edge = 200 # sampling points per edge per time step

        t_list, x_list, y_list, h_list = [], [], [], []

        for t in t_array:
            # Left boundary x=0, h=0m
            # y_left = np.random.uniform(self.domain_cfg.y_min, self.domain_cfg.y_max, n_points_per_edge)
            y_left = np.linspace(self.domain_cfg.y_min, self.domain_cfg.y_max, n_points_per_edge)
            x_left = np.zeros(n_points_per_edge)
            t_left = np.ones(n_points_per_edge) * t
            h_left = np.zeros(n_points_per_edge)

            t_list.append(t_left)
            x_list.append(x_left)
            y_list.append(y_left)
            h_list.append(h_left)

            # Right boundary x=L, h=0m
            # y_right = np.random.uniform(self.domain_cfg.y_min, self.domain_cfg.y_max, n_points_per_edge)
            y_right = np.linspace(self.domain_cfg.y_min, self.domain_cfg.y_max, n_points_per_edge)
            x_right = np.ones(n_points_per_edge) * self.domain_cfg.x_max
            t_right = np.ones(n_points_per_edge) * t
            h_right = np.zeros(n_points_per_edge)

            t_list.append(t_right)
            x_list.append(x_right)
            y_list.append(y_right)
            h_list.append(h_right)

            # Bottom boundary y=0, h=0m
            # x_bottom = np.random.uniform(self.domain_cfg.x_min, self.domain_cfg.x_max, n_points_per_edge)
            x_bottom = np.linspace(self.domain_cfg.x_min, self.domain_cfg.x_max, n_points_per_edge)
            y_bottom = np.zeros(n_points_per_edge)
            t_bottom = np.ones(n_points_per_edge) * t
            h_bottom = np.zeros(n_points_per_edge)

            t_list.append(t_bottom)
            x_list.append(x_bottom)
            y_list.append(y_bottom)
            h_list.append(h_bottom)

            # Top boundary y=H, h=0m
            # x_top = np.random.uniform(self.domain_cfg.x_min, self.domain_cfg.x_max, n_points_per_edge)
            x_top = np.linspace(self.domain_cfg.x_min, self.domain_cfg.x_max, n_points_per_edge)
            y_top = np.ones(n_points_per_edge) * self.domain_cfg.y_max
            t_top = np.ones(n_points_per_edge) * t
            h_top = np.zeros(n_points_per_edge)

            t_list.append(t_top)
            x_list.append(x_top)
            y_list.append(y_top)
            h_list.append(h_top)

        bc_data = {
            't': np.concatenate(t_list),
            'x': np.concatenate(x_list),
            'y': np.concatenate(y_list),
            'h': np.concatenate(h_list)
        }

        self.logger.info(f"  Dirichlet boundary condition points: {len(bc_data['t'])} (4 edges × {n_points_per_edge} points × {len(t_array)} time steps)")
        self.logger.info(f"  All boundaries: h=0m")
        return bc_data

    def generate_bc_neumann_data(self):
        """
        Generate Neumann boundary condition data (raw data, not dimensionless)

        Top and bottom boundaries are zero-flux boundaries: ∂h/∂y = 0
        Bottom boundary y=0, top boundary y=H

        Returns:
            bc_data: dict with keys 't', 'x', 'y'
        """
        self.logger.info("Generating Neumann boundary condition data (top/bottom zero-flux)...")

        t_array = self.generate_time_array()
        n_points_per_edge = 200  # sampling points per edge per time step

        t_list, x_list, y_list = [], [], []

        for t in t_array:
            # Bottom boundary y=0
            x_bottom = np.random.uniform(self.domain_cfg.x_min, self.domain_cfg.x_max, n_points_per_edge)
            y_bottom = np.zeros(n_points_per_edge)
            t_bottom = np.ones(n_points_per_edge) * t

            t_list.append(t_bottom)
            x_list.append(x_bottom)
            y_list.append(y_bottom)

            # Top boundary y=H
            x_top = np.random.uniform(self.domain_cfg.x_min, self.domain_cfg.x_max, n_points_per_edge)
            y_top = np.ones(n_points_per_edge) * self.domain_cfg.y_max
            t_top = np.ones(n_points_per_edge) * t

            t_list.append(t_top)
            x_list.append(x_top)
            y_list.append(y_top)

        bc_data = {
            't': np.concatenate(t_list),
            'x': np.concatenate(x_list),
            'y': np.concatenate(y_list)
        }

        self.logger.info(f"  Neumann boundary condition points: {len(bc_data['t'])} (2 edges × {n_points_per_edge} points × {len(t_array)} time steps)")
        return bc_data

    def generate_collocation_data(self):
        """
        Generate collocation point data (raw data, not dimensionless)

        Supports three spatial sampling methods (controlled by train_cfg.spatial_sampling):

        1. uniform: random spatial sampling for each time point
           - Independently sample n_coll_points spatial points at each time step

        2. constant: fixed spatial points, shared across all time points
           - Sample 1.5*n_coll_points spatial points once

        3. shifty: hybrid sampling (50% random + 50% well-refined)
           - Sampling probability formula: p(r) = exp(-r/R * factor) / Σexp(-r/R * f)
           - factor linearly decays from 20 to 1 over time

        Note: Points within 0.5m of wells are filtered out

        Returns:
            coll_data: dict with keys 't', 'x', 'y', 'K', 'Kx', 'Ky'
        """
        self.logger.info("Generating collocation point data...")
        self.logger.info(f"  Spatial sampling method: {self.train_cfg.spatial_sampling}")

        t_array = self.generate_time_array()
        cfg = self.train_cfg
        well_radius = 0.5  # filter radius

        # Define computational domain polygon
        poly = [[self.domain_cfg.x_min, self.domain_cfg.y_min],
                [self.domain_cfg.x_max, self.domain_cfg.y_min],
                [self.domain_cfg.x_max, self.domain_cfg.y_max],
                [self.domain_cfg.x_min, self.domain_cfg.y_max]]

        # Well coordinate list
        well_list = [list(pos) for pos in self.well_cfg.positions]

        t_list, x_list, y_list = [], [], []
        K_list, Kx_list, Ky_list = [], [], []

        if cfg.spatial_sampling == 'uniform':
            # ============ uniform: independent random sampling at each time step ============
            n_points = cfg.n_coll_points
            self.logger.info(f"  Points per time step: {n_points}")

            for t in t_array:
                # Random sampling within computational domain
                x_rand = np.random.uniform(self.domain_cfg.x_min, self.domain_cfg.x_max, n_points * 2)
                y_rand = np.random.uniform(self.domain_cfg.y_min, self.domain_cfg.y_max, n_points * 2)
                xy = np.stack([x_rand, y_rand], axis=-1)

                # Filter points near wells
                retain_mask = is_retain_not_close_wells(xy, well_list, well_radius)
                xy = xy[retain_mask][:n_points]

                # Interpolate to get K, Kx, Ky (using triangular mesh shape functions)
                K, Kx, Ky = self._interpolate_K(xy)

                t_list.append(np.ones(len(xy)) * t)
                x_list.append(xy[:, 0])
                y_list.append(xy[:, 1])
                K_list.append(K)
                Kx_list.append(Kx)
                Ky_list.append(Ky)

        elif cfg.spatial_sampling == 'constant':
            # ============ constant: fixed spatial points, shared across all time steps ============
            n_points = int(cfg.n_coll_points * 1.5)
            self.logger.info(f"  Fixed spatial points: {n_points}")

            # Sample spatial points once
            x_rand = np.random.uniform(self.domain_cfg.x_min, self.domain_cfg.x_max, n_points * 2)
            y_rand = np.random.uniform(self.domain_cfg.y_min, self.domain_cfg.y_max, n_points * 2)
            xy = np.stack([x_rand, y_rand], axis=-1)

            # Filter points near wells
            retain_mask = is_retain_not_close_wells(xy, well_list, well_radius)
            xy_fixed = xy[retain_mask][:n_points]

            # Interpolate to get K, Kx, Ky (fixed values, using triangular mesh shape functions)
            K_fixed, Kx_fixed, Ky_fixed = self._interpolate_K(xy_fixed)

            # Use same spatial points for all time steps
            for t in t_array:
                t_list.append(np.ones(len(xy_fixed)) * t)
                x_list.append(xy_fixed[:, 0].copy())
                y_list.append(xy_fixed[:, 1].copy())
                K_list.append(K_fixed.copy())
                Kx_list.append(Kx_fixed.copy())
                Ky_list.append(Ky_fixed.copy())

        elif cfg.spatial_sampling == 'shifty':
            # ============ shifty: hybrid sampling (random + well-refined) ============
            # Sampling strategy:
            #   - 50% through global random sampling, ensuring global coverage
            #   - 50% through LR_sampling well-refined sampling, capturing well gradients
            #
            # LR_sampling probability formula:
            #   p(r) = exp(-r/R * f) / Σexp(-r/R * f)
            #
            # factor linearly decays over time: 20 → 1.0

            n_points = cfg.n_coll_points
            n_random = n_points // 2  # half random sampling
            n_lr = n_points - n_random  # half well-refined sampling

            self.logger.info(f"  Points per time step: {n_points} (random:{n_random} + well-refined:{n_lr})")
            self.logger.info(f"  Refinement factor range: 10 → 1.0")


            # Refinement factor changes linearly with time
            factor_range = np.linspace(5, 1.5, len(t_array))

            for i, t in enumerate(t_array):

                # ---- Part 1: Random sampling (50%) ----
                x_rand = np.random.uniform(self.domain_cfg.x_min, self.domain_cfg.x_max, int(n_random * 1.1))
                y_rand = np.random.uniform(self.domain_cfg.y_min, self.domain_cfg.y_max, int(n_random * 1.1))
                xy_random = np.stack([x_rand, y_rand], axis=-1)
                # Filter points near wells
                retain_mask = is_retain_not_close_wells(xy_random, well_list, well_radius)
                xy_random = xy_random[retain_mask][:n_random]

                # ---- Part 2: LR_sampling well-refined sampling (50%) ----
                factor = factor_range[i]

                xy_lr = LR_sampling(well_list, radius=400, factor=factor,
                                   all_num_points=int(n_lr * 1.8),
                                   num_factor=max(int(100 * factor), 50))

                # Filter: keep only points inside computational domain
                in_poly_mask = is_in_poly(xy_lr, poly)
                xy_lr = xy_lr[in_poly_mask]

                # Filter: exclude points within 0.5m of wells
                retain_mask = is_retain_not_close_wells(xy_lr, well_list, well_radius)
                xy_lr = xy_lr[retain_mask][:n_lr]

                # ---- Merge both sampling parts ----
                xy = np.vstack([xy_random, xy_lr])
                np.random.shuffle(xy)  # shuffle order

                # Interpolate to get K, Kx, Ky (using triangular mesh shape functions)
                K, Kx, Ky = self._interpolate_K(xy)

                t_list.append(np.ones(len(xy)) * t)
                x_list.append(xy[:, 0])
                y_list.append(xy[:, 1])
                K_list.append(K)
                Kx_list.append(Kx)
                Ky_list.append(Ky)

        else:
            raise ValueError(f"Unknown spatial_sampling: {cfg.spatial_sampling}")

        coll_data = {
            't': np.concatenate(t_list),
            'x': np.concatenate(x_list),
            'y': np.concatenate(y_list),
            'K': np.concatenate(K_list),
            'Kx': np.concatenate(Kx_list),
            'Ky': np.concatenate(Ky_list),
            'n_time_steps': len(t_array),  # record time step count for temporal causal strategy
        }

        self.logger.info(f"  Total collocation points: {len(coll_data['t'])} ({len(t_array)} time steps)")

        # import matplotlib.pyplot as plt
        # plt.tricontourf(np.concatenate(x_list)[::10],np.concatenate(y_list)[::10],np.concatenate(Kx_list)[::10])
        # plt.colorbar()

        return coll_data

    def _interpolate_K(self, xy, return_gradient=True):
        """
        Interpolate hydraulic conductivity at spatial points using triangular mesh shape functions

        Args:
            xy: (n, 2) spatial point coordinates
            return_gradient: whether to return derivatives

        Returns:
            K, Kx, Ky or only K
        """
        return self.K_interpolator.interpolate(xy, return_gradient=return_gradient)

    def generate_observation_data(self):
        """
        Generate observation data (raw data, not dimensionless)

        1. Well observations: each well head → mapped to 12 points on 0.1m radius circle around well
           5 wells × 12 points = 60 spatial locations
        2. 4×4 observation points: replicated 4 times = 64 locations (optional)

        Returns:
            obse_data: dict with keys 't', 'x', 'y', 'h'
        """
        self.logger.info("Generating observation data...")

        cfg = self.train_cfg
        data_t_array = np.linspace(0, 2, 201)

        t_list, x_list, y_list, h_list = [], [], [], []

        # ============ 1. Well observation data ============
        # Circle sampling points per well
        r_obse = cfg.well_obse_radius  # 0.5m
        n_circle = cfg.n_well_obse_points  # 12 points
        theta = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)

        well_obse_count = 0
        for i_well in range(self.well_cfg.n_wells):
            x_well, y_well = self.well_cfg.positions[i_well]

            # 12 points on the circle
            x_circle = x_well + r_obse * np.cos(theta)
            y_circle = y_well + r_obse * np.sin(theta)

            for i_t, t in enumerate(data_t_array):
                if i_t >= self.well_head.shape[1]:
                    break

                h = self.well_head[i_well, i_t]  # head at this well at this time

                # Skip t=0, as initial condition is handled separately by IC
                if i_t > 0:
                    # 12 points share the same head value
                    for k in range(n_circle):
                        t_list.append(t)
                        x_list.append(x_circle[k])
                        y_list.append(y_circle[k])
                        h_list.append(h)
                        well_obse_count += 1

        self.logger.info(f"  Well observation data: {well_obse_count} records "
              f"({self.well_cfg.n_wells} wells × {n_circle} points × {len(data_t_array)} time steps)")

        # ============ 2. 4×4 observation point data (optional) ============
        obse_4x4_count = 0
        if cfg.use_obse_4x4:
            n_repeat = cfg.obse_4x4_repeat  # replicate 4 times

            for _ in range(n_repeat):
                for i_obs in range(self.obse_pos.shape[0]):
                    x_obs, y_obs = self.obse_pos[i_obs]

                    for i_t, t in enumerate(data_t_array):
                        if i_t >= self.obse_head.shape[1]:
                            break

                        if i_t>0:
                            h = self.obse_head[i_obs, i_t]
                            # print(h)
                            t_list.append(t)
                            x_list.append(x_obs)
                            y_list.append(y_obs)
                            h_list.append(h)
                            obse_4x4_count += 1

            self.logger.info(f"  4×4 observation data: {obse_4x4_count} records "
                  f"(16 points × {n_repeat} times × {len(data_t_array)} time steps)")
        else:
            self.logger.info(f"  4×4 observation data: not used")

        obse_data = {
            't': np.array(t_list),
            'x': np.array(x_list),
            'y': np.array(y_list),
            'h': np.array(h_list)
        }

        self.logger.info(f"  Total observation data: {len(obse_data['t'])} records")
        return obse_data

    def generate_well_flux_data(self):
        """
        Generate well flux constraint data (raw data, not dimensionless)

        Sample on 0.5m radius circle around each well, constrain flow velocity

        Returns:
            well_data: dict with keys 't', 'x', 'y', 'vx', 'vy', 'K'
        """
        self.logger.info("Generating well flux constraint data...")

        cfg = self.train_cfg
        # t_array = self.generate_time_array(n=100)
        t_array = np.linspace(2/100, 2, 100)

        r = cfg.well_flux_radius  # 0.5m
        n_circle = cfg.n_well_flux_points  # 32 points
        theta = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)

        t_list, x_list, y_list = [], [], []
        vx_list, vy_list, K_list = [], [], []

        for i_well in range(self.well_cfg.n_wells):
            x_well, y_well = self.well_cfg.positions[i_well]
            well_type = self.well_cfg.well_types[i_well]

            # Points on the circle
            x_circle = x_well + r * np.cos(theta)
            y_circle = y_well + r * np.sin(theta)

            # Use K value loaded from well_K.pkl (constant for each well)
            K_well = self.well_K[i_well, 0]
            print(f'Well {i_well+1} hydraulic conductivity: {K_well}')
            for t in t_array:

                # Calculate theoretical velocity (radial flow)
                vx, vy = self._compute_well_velocity(
                    x_well, y_well, x_circle, y_circle,
                    well_type, r
                )

                for k in range(n_circle):
                    t_list.append(t)
                    x_list.append(x_circle[k])
                    y_list.append(y_circle[k])
                    vx_list.append(vx[k])
                    vy_list.append(vy[k])
                    K_list.append(K_well)

        well_data = {
            't': np.array(t_list),
            'x': np.array(x_list),
            'y': np.array(y_list),
            'vx': np.array(vx_list),
            'vy': np.array(vy_list),
            'K': np.array(K_list),
            'n_time_steps': len(t_array),  # record time step count for temporal causal strategy
        }

        self.logger.info(f"  Well flux constraint points: {len(well_data['t'])} "
              f"({self.well_cfg.n_wells} wells × {n_circle} points × {len(t_array)} time steps)")
        return well_data

    def _get_K_at_point(self, x, y):
        """Get hydraulic conductivity at specified point (using triangular mesh interpolation)"""
        xy = np.array([[x, y]])
        K = self.K_interpolator.interpolate(xy, return_gradient=False)
        return K[0]

    def _compute_well_velocity(self, x_well, y_well, x_circle, y_circle, well_type, r):
        """
        Calculate flow velocity on circle around well (radial flow assumption)

        vr = Q / (2*pi*r*M)
        vx = vr * (x - x_well) / r
        vy = vr * (y - y_well) / r
        """
        M = self.scale_cfg.M  # 50m
        Q = 200.0  # m³/d (示例流量)

        vr = well_type * Q / (2 * np.pi * r * M)

        dx = x_circle - x_well
        dy = y_circle - y_well
        dist = np.sqrt(dx**2 + dy**2) + 1e-10

        vx = vr * dx / dist
        vy = vr * dy / dist

        return vx, vy

    def generate_test_data(self):
        """
        Generate test data (raw data, not dimensionless)

        Use 4 complete snapshots for evaluation

        Returns:
            test_data: dict with keys 'txy', 'h', 'plot_x', 'plot_y', 'times'
        """
        self.logger.info("Generating test data...")

        # Spatiotemporal coordinates and head for 4 snapshots (sample every 6 points to reduce test point count)
        test_txy = self.snap_txy[:, ::4]  # (4, n_test_points, 3)
        test_h = self.snap_data[:, 0, ::4].copy()  # (4, n_test_points)


        # Filter spatial points within 0.5m of wells
        well_radius = 0.5  # filter radius
        xy = test_txy[0, :, 1:3]  # (n_spatial, 2)

        # Calculate minimum distance from each spatial point to all wells
        min_dist_to_wells = np.full(len(xy), np.inf)
        for i_well in range(self.well_cfg.n_wells):
            well_pos = self.well_cfg.positions[i_well]
            dist = np.sqrt((xy[:, 0] - well_pos[0])**2 + (xy[:, 1] - well_pos[1])**2)
            min_dist_to_wells = np.minimum(min_dist_to_wells, dist)

        # Keep points with distance greater than well_radius
        valid_mask = min_dist_to_wells > well_radius
        test_txy = test_txy[:,valid_mask]
        test_h = test_h[:,valid_mask]
        # Spatial coordinates for plotting
        plot_x = test_txy[0, :, 1].copy()
        plot_y = test_txy[0, :, 2].copy()

        test_data = {
            'txy': test_txy,
            'h': test_h,
            'plot_x': plot_x,
            'plot_y': plot_y,
            'times': self.snap_times
        }

        self.logger.info(f"  Test data: 4 snapshots, {test_txy.shape[1]} points each")
        return test_data

    def normalize_all_data(self, raw_data_dict):
        """
        Unified dimensionless processing

        Args:
            raw_data_dict: dictionary of various raw data

        Returns:
            norm_data_dict: dictionary of dimensionless data
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Dimensionless Processing")
        self.logger.info("=" * 50)

        cfg = self.scale_cfg
        norm_data = {}

        # ============ Initial Condition ============
        ic = raw_data_dict['ic']
        norm_data['ic_inputs'] = np.stack([
            cfg.normalize_t(ic['t']),
            cfg.normalize_x(ic['x']),
            cfg.normalize_y(ic['y'])
        ], axis=-1).astype(np.float32)
        norm_data['ic_labels'] = cfg.normalize_h(ic['h']).reshape(-1, 1).astype(np.float32)
        self.logger.info(f"  ic_inputs: {norm_data['ic_inputs'].shape}")

        # ============ Dirichlet Boundary Condition ============
        diri = raw_data_dict['bc_dirichlet']
        norm_data['diri_inputs'] = np.stack([
            cfg.normalize_t(diri['t']),
            cfg.normalize_x(diri['x']),
            cfg.normalize_y(diri['y'])
        ], axis=-1).astype(np.float32)
        norm_data['diri_labels'] = cfg.normalize_h(diri['h']).reshape(-1, 1).astype(np.float32)
        self.logger.info(f"  diri_inputs: {norm_data['diri_inputs'].shape}")

        # ============ Neumann Boundary Condition ============
        if 'bc_neumann' in raw_data_dict:
            neum = raw_data_dict['bc_neumann']
            norm_data['neum_inputs'] = np.stack([
                cfg.normalize_t(neum['t']),
                cfg.normalize_x(neum['x']),
                cfg.normalize_y(neum['y'])
            ], axis=-1).astype(np.float32)
            self.logger.info(f"  neum_inputs: {norm_data['neum_inputs'].shape}")

        # ============ Collocation Data ============
        coll = raw_data_dict['collocation']
        norm_data['coll_inputs'] = np.stack([
            cfg.normalize_t(coll['t']),
            cfg.normalize_x(coll['x']),
            cfg.normalize_y(coll['y']),
            cfg.normalize_K(coll['K']),
            cfg.normalize_Kx(coll['Kx']),
            cfg.normalize_Ky(coll['Ky'])
        ], axis=-1).astype(np.float32)
        self.logger.info(f"  coll_inputs: {norm_data['coll_inputs'].shape}")

        # Temporal causal strategy: organize collocation data by time step as (n_time_steps, n_points_per_step, features)
        n_time_steps_coll = coll.get('n_time_steps', self.train_cfg.n_time_steps)
        n_total_coll = len(coll['t'])
        n_points_per_step_coll = n_total_coll // n_time_steps_coll
        norm_data['coll_inputs_by_time'] = norm_data['coll_inputs'].reshape(
            n_time_steps_coll, n_points_per_step_coll, -1
        )
        norm_data['n_time_steps_coll'] = n_time_steps_coll
        self.logger.info(f"  coll_inputs_by_time: {norm_data['coll_inputs_by_time'].shape} "
                        f"(for temporal causal strategy)")

        # ============ Observation Data ============
        obse = raw_data_dict['observation']
        norm_data['obse_inputs'] = np.stack([
            cfg.normalize_t(obse['t']),
            cfg.normalize_x(obse['x']),
            cfg.normalize_y(obse['y'])
        ], axis=-1).astype(np.float32)
        norm_data['obse_labels'] = cfg.normalize_h(obse['h']).reshape(-1, 1).astype(np.float32)
        self.logger.info(f"  obse_inputs: {norm_data['obse_inputs'].shape}")

        # ============ Well Flux Constraint Data ============
        well = raw_data_dict['well_flux']
        norm_data['well_inputs'] = np.stack([
            cfg.normalize_t(well['t']),
            cfg.normalize_x(well['x']),
            cfg.normalize_y(well['y'])
        ], axis=-1).astype(np.float32)
        # Velocity keeps original dimensions, handled in loss function
        norm_data['well_labels'] = np.stack([
            well['vx'],
            well['vy']
        ], axis=-1).astype(np.float32)
        norm_data['well_K'] = cfg.normalize_K(well['K']).reshape(-1, 1).astype(np.float32)
        self.logger.info(f"  well_inputs: {norm_data['well_inputs'].shape}")

        # Temporal causal strategy: organize well constraint data by time step as (n_time_steps, n_points_per_step, features)
        n_time_steps_well = well.get('n_time_steps', 100)  # well constraint defaults to 100 time steps
        n_total_well = len(well['t'])
        n_points_per_step_well = n_total_well // n_time_steps_well
        norm_data['well_inputs_by_time'] = norm_data['well_inputs'].reshape(
            n_time_steps_well, n_points_per_step_well, -1
        )
        norm_data['well_labels_by_time'] = norm_data['well_labels'].reshape(
            n_time_steps_well, n_points_per_step_well, -1
        )
        norm_data['well_K_by_time'] = norm_data['well_K'].reshape(
            n_time_steps_well, n_points_per_step_well, -1
        )
        norm_data['n_time_steps_well'] = n_time_steps_well
        self.logger.info(f"  well_inputs_by_time: {norm_data['well_inputs_by_time'].shape} "
                        f"(for temporal causal strategy)")

        # ============ Test Data ============
        test = raw_data_dict['test']
        test_txy_norm = []
        for i in range(test['txy'].shape[0]):
            txy = test['txy'][i].copy()
            txy_norm = np.stack([
                cfg.normalize_t(txy[:, 0]),
                cfg.normalize_x(txy[:, 1]),
                cfg.normalize_y(txy[:, 2])
            ], axis=-1)
            test_txy_norm.append(txy_norm)
        norm_data['test_txy'] = np.stack(test_txy_norm, axis=0).astype(np.float32)
        norm_data['test_head'] = cfg.normalize_h(test['h']).astype(np.float32)
        norm_data['plot_x'] = test['plot_x']
        norm_data['plot_y'] = test['plot_y']
        norm_data['test_times'] = test['times']
        self.logger.info(f"  test_txy: {norm_data['test_txy'].shape}")

        return norm_data

    def prepare_all_data(self, save_path=None):
        """
        Prepare all training data

        Process:
        1. Generate various raw data (not dimensionless)
        2. Unified dimensionless processing
        3. Return processed data dictionary

        Returns:
            data_dict: dictionary containing all dimensionless data
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Generating Raw Training Data")
        self.logger.info("=" * 50)

        # ============ 1. Generate various raw data ============
        raw_data = {
            'ic': self.generate_ic_data(),
            'bc_dirichlet': self.generate_bc_dirichlet_data(),
            'collocation': self.generate_collocation_data(),
            'observation': self.generate_observation_data(),
            'well_flux': self.generate_well_flux_data(),
            'test': self.generate_test_data()
        }

        # ============ 2. Unified dimensionless processing ============
        norm_data = self.normalize_all_data(raw_data)

        # Add configuration information
        norm_data['scale_config'] = self.scale_cfg
        norm_data['train_config'] = self.train_cfg

        # ============ 3. Save ============
        if save_path:
            # Save raw data
            # write_pkl(raw_data, save_path.replace('.pkl', '_raw.pkl'))
            # Save dimensionless data
            # write_pkl(norm_data, save_path)
            self.logger.info(f"\nData saved to: {save_path}")

        return norm_data

    def to_torch(self, data_dict):
        """Convert numpy data to torch tensor and move to GPU"""
        torch_dict = {}

        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                torch_dict[key] = torch.from_numpy(value).float().to(device)
            else:
                torch_dict[key] = value

        return torch_dict


# ============== Convenience Functions ==============
def load_and_prepare_data(data_path, train_cfg=None, save_path=None):
    """
    Convenience function to load and prepare all data

    Args:
        data_path: raw data path
        train_cfg: training configuration (optional)
        save_path: save path (optional)

    Returns:
        data_dict: data dictionary in torch tensor format
        scale_cfg: scaling configuration
    """
    processor = DataProcessor(data_path, train_cfg=train_cfg)
    data_dict = processor.prepare_all_data(save_path)
    return processor.to_torch(data_dict), processor.scale_cfg


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Test data processing
    data_path = '../pre_pkl_data/'

    # Test different time sampling methods
    for sampling in ['uniform', 'lg', 'ln']:
        module_logger.info(f"\n{'='*60}")
        module_logger.info(f"Testing time sampling method: {sampling}")
        module_logger.info('='*60)

        train_cfg = TrainingConfig()
        train_cfg.time_sampling = sampling
        train_cfg.use_obse_4x4 = True

        processor = DataProcessor(data_path, train_cfg=train_cfg)

        # View time sampling results
        t_array = processor.generate_time_array()
        module_logger.info(f"Time sampling ({sampling}):")
        module_logger.info(f"  First 5: {t_array[:5]}")
        module_logger.info(f"  Last 5: {t_array[-5:]}")

    # Complete data processing test
    module_logger.info(f"\n{'='*60}")
    module_logger.info("Complete Data Processing Test")
    module_logger.info('='*60)

    train_cfg = TrainingConfig()
    train_cfg.time_sampling = 'lg'
    train_cfg.use_obse_4x4 = True

    processor = DataProcessor(data_path, train_cfg=train_cfg)
    data_dict = processor.prepare_all_data(save_path='./processed_data.pkl')

    module_logger.info("\nData Statistics:")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            module_logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
