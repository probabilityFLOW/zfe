# Graph_loader.py
# -*- coding: utf-8 -*-

import os
import math
import torch
import matplotlib
matplotlib.use("Agg")  # headless (Linux server safe)
import matplotlib.pyplot as plt


class GraphLoader:
    """
    Synthetic Gaussian data loader for structured graphical models.

    Generates x ~ N(0, Sigma) where Theta = Sigma^{-1} is a structured precision matrix:
      - higher-order Markov chain (banded)
      - 2D lattice grid (4-neighbor or 8-neighbor)

    Also saves heatmaps (Theta and Sigma) to disk.
    """

    def __init__(
        self,
        batch_size: int = 32,
        dim: int = 64,
        device: str = "cuda",
        graph_type: str = "markov_k",   # "markov_k" or "grid2d"
        # markov_k params
        markov_order: int = 2,          # k in k-th order chain
        chain_weights=None,             # list/tuple length k, e.g. [0.6, 0.3]
        # grid2d params
        grid_shape=None,                # (H, W); if None, inferred from dim as square-ish
        grid_connectivity: str = "4",   # "4" or "8"
        grid_weight: float = 1.0,
        diag_strength: float = 0.5,     # extra diagonal to ensure SPD margin
        n_train: int = 50_000,
        n_test: int = 10_000,
        seed: int = 0,
        save_dir: str = "results_graph_loader",
        save_prefix: str = "toygraph",
        dtype: torch.dtype = torch.float32,
        mask_mode: str = "chain_adjacent",  # "chain_adjacent" or "grid_block"
        mask_block_size: int = 2,           # used for grid_block
    ):
        self.batch_size = int(batch_size)
        self.dim = int(dim)
        self.device = device
        self.graph_type = graph_type
        self.markov_order = int(markov_order)
        self.grid_connectivity = str(grid_connectivity)
        self.grid_weight = float(grid_weight)
        self.diag_strength = float(diag_strength)
        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.seed = int(seed)
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.dtype = dtype
        self.mask_mode = mask_mode
        self.mask_block_size = int(mask_block_size)

        os.makedirs(self.save_dir, exist_ok=True)

        # Reproducibility
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        self._cpu_gen = g

        # Build Theta (precision) then Sigma, then sample datasets
        if self.graph_type == "markov_k":
            Theta = self._build_markov_k_theta(
                dim=self.dim,
                k=self.markov_order,
                weights=chain_weights,
                diag_strength=self.diag_strength,
            )
        elif self.graph_type == "grid2d":
            Theta = self._build_grid2d_theta(
                dim=self.dim,
                grid_shape=grid_shape,
                connectivity=self.grid_connectivity,
                weight=self.grid_weight,
                diag_strength=self.diag_strength,
            )
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

        self.Theta = Theta.to(device=self.device, dtype=self.dtype)
        # Convert to covariance
        self.Sigma = torch.linalg.inv(self.Theta)

        # Save heatmaps
        self._save_heatmaps(self.Theta, self.Sigma)

        # Sample data
        # Note: MultivariateNormal wants CPU generator? We'll sample on device directly without passing generator.
        mvn = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.dim, device=self.device, dtype=self.dtype),
            covariance_matrix=self.Sigma
        )

        self.xdata = mvn.sample((self.n_train,))
        self.xtestdata = mvn.sample((self.n_test,))

        print(
            f"[GraphLoader] graph_type={self.graph_type} dim={self.dim} "
            f"batch_size={self.batch_size} device={self.device} "
            f"train={self.n_train} test={self.n_test} savedir={self.save_dir}"
        )

    # -------------------------
    # Graph constructions
    # -------------------------

    def _build_markov_k_theta(self, dim: int, k: int, weights=None, diag_strength: float = 0.5):
        """
        Higher-order Markov chain precision (banded SPD matrix).

        We connect i <-> i+d for d=1..k with specified weights (positive off-diagonals),
        then enforce strict diagonal dominance to guarantee SPD.

        Note: In Gaussian graphical models, sparsity pattern of Theta encodes conditional independencies.
        """
        if k < 1:
            raise ValueError("markov_order k must be >= 1")

        if weights is None:
            # Default: decreasing weights
            # Example k=3 -> [0.8, 0.4, 0.2]
            weights = [0.8 * (0.5 ** (d - 1)) for d in range(1, k + 1)]
        else:
            if len(weights) != k:
                raise ValueError(f"chain_weights must have length {k}, got {len(weights)}")
            weights = [float(w) for w in weights]

        Theta = torch.zeros((dim, dim), dtype=torch.float64)

        # Fill banded structure
        for d in range(1, k + 1):
            w = weights[d - 1]
            for i in range(dim - d):
                Theta[i, i + d] = w
                Theta[i + d, i] = w

        # Enforce diagonal dominance: Theta[ii] = sum_j |Theta_ij| + diag_strength
        abs_row_sum = torch.sum(torch.abs(Theta), dim=1)
        Theta.fill_diagonal_(0.0)
        Theta = Theta + torch.diag(abs_row_sum + diag_strength)

        # Slightly strengthen endpoints (optional)
        Theta[0, 0] += 0.25 * diag_strength
        Theta[-1, -1] += 0.25 * diag_strength

        return Theta

    def _build_grid2d_theta(
        self,
        dim: int,
        grid_shape=None,
        connectivity: str = "4",
        weight: float = 1.0,
        diag_strength: float = 0.5,
    ):
        """
        2D lattice grid precision matrix.

        If grid_shape is None, we infer a near-square grid (H, W) s.t. H*W = dim.
        If dim is not factorizable into a rectangle nicely, we raise an error.

        connectivity:
          - "4": N,S,E,W
          - "8": also include diagonals
        """
        if grid_shape is None:
            # Infer near-square factors
            h = int(math.sqrt(dim))
            while h > 1 and dim % h != 0:
                h -= 1
            w = dim // h
            if h * w != dim:
                raise ValueError(f"Cannot infer rectangular grid for dim={dim}. Provide grid_shape=(H,W).")
            H, W = h, w
        else:
            H, W = grid_shape
            if H * W != dim:
                raise ValueError(f"grid_shape {grid_shape} does not match dim={dim} (H*W must equal dim).")

        conn8 = (connectivity == "8")
        if connectivity not in ("4", "8"):
            raise ValueError("grid_connectivity must be '4' or '8'")

        Theta = torch.zeros((dim, dim), dtype=torch.float64)

        def idx(r, c):
            return r * W + c

        # Neighbor offsets
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if conn8:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Add edges
        for r in range(H):
            for c in range(W):
                i = idx(r, c)
                for dr, dc in offsets:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        j = idx(rr, cc)
                        Theta[i, j] = weight
                        Theta[j, i] = weight

        # Enforce diagonal dominance for SPD
        abs_row_sum = torch.sum(torch.abs(Theta), dim=1)
        Theta.fill_diagonal_(0.0)
        Theta = Theta + torch.diag(abs_row_sum + diag_strength)

        return Theta

    # -------------------------
    # Heatmap saving
    # -------------------------

    def _save_heatmaps(self, Theta: torch.Tensor, Sigma: torch.Tensor):
        """
        Save heatmaps for Theta (precision) and Sigma (covariance).
        Stored as PNG files in save_dir.
        """
        # Move to CPU for plotting
        Theta_cpu = Theta.detach().cpu().to(torch.float32).numpy()
        Sigma_cpu = Sigma.detach().cpu().to(torch.float32).numpy()

        theta_path = os.path.join(self.save_dir, f"{self.save_prefix}_Theta_heatmap.png")
        sigma_path = os.path.join(self.save_dir, f"{self.save_prefix}_Sigma_heatmap.png")

        self._plot_heatmap(Theta_cpu, "Precision matrix Θ (sparsity shows graph)", theta_path)
        self._plot_heatmap(Sigma_cpu, "Covariance matrix Σ", sigma_path)

        print(f"[GraphLoader] Saved heatmaps:\n  - {theta_path}\n  - {sigma_path}")

    def _plot_heatmap(self, M, title: str, path: str):
        plt.figure(figsize=(6, 5), dpi=160)
        plt.imshow(M, aspect="auto")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    # -------------------------
    # Masking logic (similar spirit to your ToyLoader)
    # -------------------------

    def _process_batch(self, batch1: torch.Tensor, batch2: torch.Tensor):
        if self.mask_mode == "chain_adjacent":
            # exactly your original: select 1 dim, maybe also the next
            indices = torch.randint(0, self.dim, (batch1.size(0),), device=self.device)
            mask = torch.zeros((batch1.size(0), self.dim), device=self.device, dtype=batch1.dtype)
            mask.scatter_(1, indices.unsqueeze(1), 1.0)

            should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (indices < self.dim - 1)
            mask[torch.arange(batch1.size(0), device=self.device)[should_mask_next], indices[should_mask_next] + 1] = 1.0
            return batch1, batch2, mask

        elif self.mask_mode == "grid_block":
            # Mask a contiguous block in the flattened grid (requires grid2d to be meaningful).
            # We treat indices as 2D if dim is a rectangle; otherwise this is still a block in 1D view.
            B = batch1.size(0)
            mask = torch.zeros((B, self.dim), device=self.device, dtype=batch1.dtype)

            # Pick random starting index; mark a short block of length mask_block_size (clipped)
            start = torch.randint(0, self.dim, (B,), device=self.device)
            for b in range(B):
                s = int(start[b].item())
                e = min(self.dim, s + self.mask_block_size)
                mask[b, s:e] = 1.0
            return batch1, batch2, mask

        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

    def get_batch(self):
        idx1 = torch.randint(0, self.xdata.size(0), (self.batch_size,), device=self.device)
        idx2 = torch.randint(0, self.xdata.size(0), (self.batch_size,), device=self.device)
        batch1 = self.xdata[idx1]
        batch2 = self.xdata[idx2]
        return self._process_batch(batch1, batch2)

    def get_test_batch(self):
        idx1 = torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)
        idx2 = torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)
        batch1 = self.xtestdata[idx1]
        batch2 = self.xtestdata[idx2]
        return self._process_batch(batch1, batch2)


# -------------------------
# Example usage (writes heatmaps to disk)
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example A: Higher-order Markov chain (k=3)
    loader_chain = GraphLoader(
        batch_size=64,
        dim=50,
        device=device,
        graph_type="markov_k",
        markov_order=3,
        chain_weights=[0.8, 0.4, 0.2],
        diag_strength=0.5,
        save_dir="results_graph_loader",
        save_prefix="markov3_dim50",
        mask_mode="chain_adjacent",
    )
    v1, v2, mask = loader_chain.get_batch()
    print("[markov_k] batch shapes:", v1.shape, v2.shape, mask.shape)

    # Example B: 2D grid (10x10 = 100 dims), 4-neighbor connectivity
    loader_grid = GraphLoader(
        batch_size=64,
        dim=100,
        device=device,
        graph_type="grid2d",
        grid_shape=(10, 10),
        grid_connectivity="4",
        grid_weight=1.0,
        diag_strength=0.5,
        save_dir="results_graph_loader",
        save_prefix="grid10x10_conn4",
        mask_mode="grid_block",
        mask_block_size=5,
    )
    v1g, v2g, maskg = loader_grid.get_batch()
    print("[grid2d] batch shapes:", v1g.shape, v2g.shape, maskg.shape)
