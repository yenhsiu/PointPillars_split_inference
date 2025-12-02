# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vector Quantization (VQ) and Residual Quantization (RQ) Implementation with Progressive Training Support
"""

from typing import Iterable
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class VQEmbedding(nn.Embedding):
    """
    Vector Quantization Embedding with progressive training support.
    
    Supports:
    - EMA updates for codebook learning
    - Progressive training with configurable active/frozen ranges
    - Gradient-based training with automatic frozen embedding protection
    - Automatic restart of unused codebook entries
    
    Args:
        n_embed (int): Number of embeddings in codebook
        embed_dim (int): Dimension of each embedding vector
        ema (bool): Whether to use EMA for codebook updates
        decay (float): EMA decay factor (default: 0.99)
        restart_unused_codes (bool): Whether to reinitialize unused codes
        eps (float): Small constant for numerical stability
    """

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        self._trainable = True
        self.active_n_embed = n_embed
        self.frozen_n_embed = 0
        self._use_ema = ema
        self._hook_handle = None
        
        # Always register EMA buffers (needed for both modes)
        self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
        self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())
        # Snapshot of cluster_size_ema for frozen embeddings (for stable weight computation)
        self.register_buffer('frozen_cluster_size_ema', torch.zeros(n_embed))
        
        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
        else:
            self._register_gradient_hook()

    @property
    def trainable(self):
        """Returns whether the codebook is trainable."""
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        """Sets the trainable state and updates weight gradient requirement."""
        self._trainable = value
        self.weight.requires_grad = (not self.use_ema) and self._trainable

    @property
    def use_ema(self):
        """Returns whether EMA updates are enabled."""
        return self._use_ema

    @use_ema.setter
    def use_ema(self, value):
        """Sets EMA mode and manages gradient hooks accordingly."""
        self._use_ema = value
        self.weight.requires_grad = (not self._use_ema) and self._trainable
        
        if not value and self._trainable:
            self._register_gradient_hook()
        else:
            self._remove_gradient_hook()

    def reset_usage_tracking(self):
        """Resets the cluster usage tracking buffer."""
        self.register_buffer('cluster_size', torch.zeros(self.active_n_embed))

    def set_frozen_n_embed(self, n):
        """
        Sets the number of frozen (non-trainable) embeddings.
        Saves a snapshot of cluster_size_ema and weights for the frozen range.
        
        Args:
            n (int): Number of embeddings to freeze from index 0
        """
        assert 0 <= n <= self.n_embed, f"frozen_n_embed must be in [0, {self.n_embed}]"
        
        # Save snapshot of cluster_size_ema for frozen range
        if n > self.frozen_n_embed:
            self.frozen_cluster_size_ema[:n] = self.cluster_size_ema[:n].clone()
        
        self.frozen_n_embed = n
        
        # Save frozen weights snapshot (for protection against weight decay)
        if n > 0:
            if not hasattr(self, '_frozen_weight_backup'):
                self._frozen_weight_backup = torch.zeros_like(self.weight[:n])
            elif self._frozen_weight_backup.shape[0] < n:
                self._frozen_weight_backup = torch.zeros_like(self.weight[:n])
            self._frozen_weight_backup[:n] = self.weight.data[:n].clone()
    
    def restore_frozen_weights(self):
        """
        Restores frozen embeddings from backup.
        This protects against weight decay in optimizers like AdamW.
        Should be called after optimizer.step() during gradient-based training.
        """
        if self.frozen_n_embed > 0 and hasattr(self, '_frozen_weight_backup'):
            self.weight.data[:self.frozen_n_embed] = self._frozen_weight_backup[:self.frozen_n_embed]

    def set_active_n_embed(self, n):
        """
        Sets the number of active embeddings to use.
        
        Args:
            n (int): Number of active embeddings
        """
        assert 0 <= n <= self.n_embed, f"active_n_embed must be in [0, {self.n_embed}]"
        assert n >= self.frozen_n_embed, f"active_n_embed ({n}) must be >= frozen_n_embed ({self.frozen_n_embed})"
        self.active_n_embed = n

    @property
    def trainable_range(self):
        """Returns the trainable range as (start_idx, end_idx)."""
        return (self.frozen_n_embed, self.active_n_embed)
    
    @property
    def n_trainable(self):
        """Returns the number of trainable embeddings."""
        return max(0, self.active_n_embed - self.frozen_n_embed)

    def _register_gradient_hook(self):
        """Register gradient hook to protect frozen and inactive embeddings."""
        if self._hook_handle is not None:
            return
        
        def gradient_hook(grad):
            """Zero out gradients for frozen and inactive embedding ranges."""
            if grad is None:
                return None
            
            grad = grad.clone()
            
            if self.frozen_n_embed > 0:
                grad[:self.frozen_n_embed] = 0.0
            
            if self.active_n_embed < self.n_embed:
                grad[self.active_n_embed:self.n_embed] = 0.0
            
            return grad
        
        self._hook_handle = self.weight.register_hook(gradient_hook)
    
    def _remove_gradient_hook(self):
        """Remove the registered gradient hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def compute_distances(self, inputs):
        """
        Computes L2 distances between input vectors and codebook embeddings.
        
        Args:
            inputs (Tensor): Input vectors of shape (..., embed_dim)
            
        Returns:
            Tensor: Distances of shape (..., active_n_embed)
        """
        if self.active_n_embed == 0:
            return torch.full((*inputs.shape[:-1], 1), float('inf'), device=inputs.device)
            
        codebook_t = self.weight[:self.active_n_embed, :].t()
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim, f"Input dim {inputs_shape[-1]} != embed_dim {embed_dim}"
        
        inputs_flat = inputs.reshape(-1, embed_dim)
        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        """
        Finds the nearest codebook embedding indices for input vectors.
        
        Args:
            inputs (Tensor): Input vectors of shape (..., embed_dim)
            
        Returns:
            Tensor: Embedding indices of shape (...)
        """
        if self.active_n_embed == 0:
            return torch.zeros(*inputs.shape[:-1], dtype=torch.long, device=inputs.device)
            
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)
        embed_idxs = embed_idxs.clamp(0, self.active_n_embed - 1)
        
        return embed_idxs

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        """
        Update EMA buffers for trainable embeddings only.
        Filters out frozen embeddings [0:frozen_n_embed].
        
        Args:
            vectors (Tensor): Input vectors of shape (..., embed_dim)
            idxs (Tensor): Assigned embedding indices of shape (...)
        """
        if not self._trainable:
            return

        n_active = self.active_n_embed
        update_start = self.frozen_n_embed
        
        if n_active == 0 or update_start >= n_active:
            return

        embed_dim = self.weight.shape[-1]

        vectors_flat = vectors.reshape(-1, embed_dim)
        idxs_flat = idxs.reshape(-1).clamp(0, n_active - 1)
        
        # Filter to trainable range only
        valid_mask = (idxs_flat >= update_start) & (idxs_flat < n_active)
        valid_idxs = idxs_flat[valid_mask]
        valid_vectors = vectors_flat[valid_mask]
        
        if len(valid_idxs) == 0:
            return
        
        # Remap to [0, n_trainable) range
        n_trainable = n_active - update_start
        remapped_idxs = valid_idxs - update_start
        
        # Compute cluster statistics
        one_hot = F.one_hot(remapped_idxs, num_classes=n_trainable).float()
        cluster_size = one_hot.sum(dim=0)
        vectors_sum_per_cluster = one_hot.t() @ valid_vectors
        
        # Update EMA buffers for trainable range
        self.cluster_size_ema[update_start:n_active].mul_(self.decay).add_(
            cluster_size, alpha=1 - self.decay
        )
        self.embed_ema[update_start:n_active].mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1 - self.decay
        )
        
        # Restart unused codes in trainable range
        if self.restart_unused_codes:
            trainable_cluster_size = self.cluster_size_ema[update_start:n_active]
            unused_indices_relative = torch.where(trainable_cluster_size < self.eps)[0]
            
            if len(unused_indices_relative) > 0 and valid_vectors.shape[0] > 0:
                unused_indices = unused_indices_relative + update_start
                n_unused = len(unused_indices)
                n_valid = valid_vectors.shape[0]
                
                if n_valid <= 0:
                    return
                
                # Randomly select vectors for reinitialization
                # WORKAROUND: PyTorch 1.10.x has torch.randperm bugs on CUDA
                # Generate indices on CPU then move to GPU
                if n_valid < n_unused:
                    # Need to sample with replacement
                    rand_indices = torch.randint(0, n_valid, (n_unused,), device=vectors.device)
                else:
                    # Generate on CPU to avoid PyTorch 1.10.x CUDA bug
                    rand_indices_cpu = torch.randperm(n_valid)[:n_unused]
                    rand_indices = rand_indices_cpu.to(vectors.device)
                
                random_vectors = valid_vectors[rand_indices]
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                self.embed_ema[unused_indices] = random_vectors
                self.cluster_size_ema[unused_indices] = 1.0 / world_size

    @torch.no_grad()
    def _update_embedding(self):
        """
        Update embedding weights from EMA buffers.
        - Frozen range [0:frozen_n_embed]: NOT UPDATED (completely frozen)
        - Trainable range [frozen_n_embed:active_n_embed]: Uses current cluster_size_ema
        """
        if not self._trainable:
            return
        
        n_active = self.active_n_embed
        frozen_end = self.frozen_n_embed
        
        if n_active == 0:
            return

        # ✅ 修復：Frozen range 完全不更新
        # 不應該重新計算 frozen embeddings，即使使用 snapshot 也會導致數值變化

        # Update trainable range with current statistics
        if frozen_end < n_active:
            trainable_cluster_size = self.cluster_size_ema[frozen_end:n_active]
            n_trainable_total = trainable_cluster_size.sum()
            n_trainable = n_active - frozen_end
            
            if n_trainable_total > 0:
                normalized_trainable = (
                    n_trainable_total * (trainable_cluster_size + self.eps) / 
                    (n_trainable_total + n_trainable * self.eps)
                ).clamp(min=self.eps)
                
                self.weight.data[frozen_end:n_active, :] = (
                    self.embed_ema[frozen_end:n_active] / normalized_trainable.unsqueeze(-1)
                )

    @torch.no_grad()
    def sync_ema_weights(self):
        """
        Manually synchronize EMA buffers to weights.
        - Frozen range: Uses frozen_cluster_size_ema snapshot
        - Trainable range: Uses current cluster_size_ema
        
        Note: Usually not needed as forward() auto-syncs after each batch.
        Use only for manual sync outside training or before saving checkpoints.
        """
        if not self.ema or not hasattr(self, 'embed_ema'):
            return

        n_embed = self.active_n_embed
        frozen_end = self.frozen_n_embed
        
        if n_embed == 0:
            return
        
        # ✅ 修復：Frozen range 完全不更新
        # 不應該重新計算 frozen embeddings
        
        # Sync trainable range with current statistics
        if frozen_end < n_embed:
            trainable_cluster_size = self.cluster_size_ema[frozen_end:n_embed]
            n_trainable_total = trainable_cluster_size.sum()
            n_trainable = n_embed - frozen_end
            
            if n_trainable_total > 0:
                normalized_trainable = (
                    n_trainable_total * (trainable_cluster_size + self.eps) / 
                    (n_trainable_total + n_trainable * self.eps)
                ).clamp(min=self.eps)

                self.weight.data[frozen_end:n_embed, :] = (
                    self.embed_ema[frozen_end:n_embed] / normalized_trainable.unsqueeze(-1)
                )

    def forward(self, inputs):
        """
        Quantize inputs to nearest codebook embeddings.
        
        Args:
            inputs (Tensor): Input vectors (..., embed_dim)
            
        Returns:
            tuple: (embeds, embed_idxs) - Quantized embeddings and their indices
        """
        embed_idxs = self.find_nearest_embedding(inputs)
        
        if self.training and self._use_ema and self._trainable:
            self._update_buffers(inputs, embed_idxs)
        
        embeds = self.embed(embed_idxs)
        
        if self.training and self._use_ema and self._trainable:
            self._update_embedding()
            
        return embeds, embed_idxs

    def embed(self, idxs):
        """
        Converts embedding indices to embedding vectors.
        
        Args:
            idxs (Tensor): Embedding indices
            
        Returns:
            Tensor: Embedding vectors
        """
        embeds = super().forward(idxs)
        return embeds

    def get_codebook_stats(self):
        """Get codebook statistics for monitoring."""
        stats = {
            'total_n_embed': self.n_embed,
            'active_n_embed': self.active_n_embed,
            'frozen_n_embed': self.frozen_n_embed,
            'trainable_n_embed': self.n_trainable,
            'trainable': self._trainable,
            'use_ema': self._use_ema,
            'weight_requires_grad': self.weight.requires_grad,
        }
        
        if hasattr(self, 'cluster_size_ema'):
            cluster_size = self.cluster_size_ema
            stats.update({
                'cluster_size_mean': cluster_size.mean().item(),
                'cluster_size_min': cluster_size.min().item(),
                'cluster_size_max': cluster_size.max().item(),
                'n_unused_codes': (cluster_size < self.eps).sum().item(),
            })
        
        return stats

    def __repr__(self):
        return (f"VQEmbedding(n_embed={self.n_embed}, embed_dim={self.embedding_dim}, "
                f"active={self.active_n_embed}, frozen={self.frozen_n_embed}, "
                f"trainable={self._trainable}, ema={self._use_ema})")


class RQBottleneck(nn.Module):
    """
    Residual Quantization Bottleneck Module with progressive training support.
    
    This module applies multiple stages of vector quantization to the residual
    from previous stages, enabling hierarchical compression of features.
    
    Args:
        latent_shape (tuple): Shape of latent features (H, W, C)
        code_shape (tuple): Shape of quantized codes (h, w, depth)
        n_embed (int or list): Number of embeddings per codebook
        decay (float or list): EMA decay factor(s)
        shared_codebook (bool): Whether to share codebook across depth
        restart_unused_codes (bool): Whether to restart unused codes
        commitment_loss (str): Type of commitment loss ('cumsum' or other)
        codebook_num (int): Number of codebooks (for compatibility)
        ema (bool): Whether to use EMA updates
    """

    def __init__(self,
                 latent_shape,
                 code_shape,
                 n_embed,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 commitment_loss='cumsum',
                 ema=True,
                 ):
        super().__init__()

        if not len(code_shape) == len(latent_shape) == 3:
            raise ValueError("incompatible code shape or latent shape")
        if any([y % x != 0 for x, y in zip(code_shape[:2], latent_shape[:2])]):
            raise ValueError("incompatible code shape or latent shape")

        embed_dim = np.prod(latent_shape[:2]) // np.prod(code_shape[:2]) * latent_shape[2]

        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)
        self.shape_divisor = torch.Size([latent_shape[i] // code_shape[i] for i in range(len(latent_shape))])
        self.shared_codebook = shared_codebook
        
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible with list types of momentums or sizes")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, (list, tuple)) else [n_embed] * self.code_shape[-1]
        self.decay = decay if isinstance(decay, (list, tuple)) else [decay] * self.code_shape[-1]
        self.ema = ema
        assert len(self.n_embed) == self.code_shape[-1]
        assert len(self.decay) == self.code_shape[-1]

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0], 
                                    embed_dim, 
                                    decay=self.decay[0], 
                                    restart_unused_codes=restart_unused_codes,
                                    ema=self.ema
                                    )
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.code_shape[-1])])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx], 
                                     embed_dim, 
                                     decay=self.decay[idx], 
                                     restart_unused_codes=restart_unused_codes,
                                     ema=self.ema
                                     ) for idx in range(self.code_shape[-1])]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss
        
    def set_ema_mode(self, ema_mode):
        """Sets EMA mode for all codebooks."""
        for cb in self.codebooks:
            cb.use_ema = ema_mode

    def to_code_shape(self, x):
        """Reshapes latent tensor to code shape."""
        (B, H, W, D) = x.shape
        (rH, rW, _) = self.shape_divisor
        x = x.reshape(B, H//rH, rH, W//rW, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H//rH, W//rW, -1)
        return x

    def to_latent_shape(self, x):
        """Reshapes code tensor to latent shape."""
        (B, h, w, _) = x.shape
        (_, _, D) = self.latent_shape
        (rH, rW, _) = self.shape_divisor
        x = x.reshape(B, h, w, rH, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h*rH, w*rW, D)
        return x

    def set_training_stage(self, active_codebook_idx, active_embed_size, full_embed_size, prev_embed_size=0):
        """
        Configures the model for a specific progressive training stage.
        
        This method sets up the codebooks for progressive training where:
        - Codebooks before active_codebook_idx are fully trained and frozen
        - The active codebook is being trained with possible frozen embeddings
        - Codebooks after active_codebook_idx are inactive
        
        Example:
            # Train codebook 0, embeddings [0:64]
            rq.set_training_stage(active_codebook_idx=0, active_embed_size=64, 
                                  full_embed_size=256, prev_embed_size=0)
            
            # Train codebook 0, embeddings [64:128], freeze [0:64]
            rq.set_training_stage(active_codebook_idx=0, active_embed_size=128,
                                  full_embed_size=256, prev_embed_size=64)

        Args:
            active_codebook_idx (int): Index of the codebook being trained (0-indexed)
            active_embed_size (int): Number of embeddings to use in current codebook
            full_embed_size (int): Total number of embeddings when codebook is fully trained
            prev_embed_size (int): Number of embeddings to freeze (from previous stage)
        
        Raises:
            AssertionError: If active_codebook_idx is out of range
        """
        assert 0 <= active_codebook_idx < len(self.codebooks), \
            f"active_codebook_idx {active_codebook_idx} out of range [0, {len(self.codebooks)})"
        
        for i, cb in enumerate(self.codebooks):
            if i == active_codebook_idx:
                # Currently training codebook
                cb.trainable = True
                cb.set_active_n_embed(active_embed_size)
                cb.set_frozen_n_embed(prev_embed_size)
            elif i < active_codebook_idx:
                # ✅ 選項 B：所有 codebook 使用相同的 embedding 數量
                # 已訓練完成的 codebook 使用當前階段的 embedding 數量，全部凍結
                cb.trainable = False
                cb.set_active_n_embed(active_embed_size)    # 使用當前階段的 embedding 數量
                cb.set_frozen_n_embed(active_embed_size)    # 全部凍結
            else:
                # Not yet trained codebooks: inactive
                cb.trainable = False
                cb.set_active_n_embed(0)
                cb.set_frozen_n_embed(0)

    def set_evaluation_stage(self, num_codebooks, num_embeddings):
        """
        Configures codebooks for evaluation mode.
        
        Sets all codebooks to non-trainable and activates only the specified
        number of codebooks with the given number of embeddings.
        
        Example:
            # Use first 2 codebooks with 256 embeddings each
            rq.set_evaluation_stage(num_codebooks=2, num_embeddings=256)
        
        Args:
            num_codebooks (int): Number of codebooks to use (from index 0)
            num_embeddings (int): Number of embeddings per active codebook
            
        Raises:
            AssertionError: If num_codebooks is out of range
        """
        assert 0 <= num_codebooks <= len(self.codebooks), \
            f"num_codebooks {num_codebooks} out of range [0, {len(self.codebooks)}]"
        
        for i, cb in enumerate(self.codebooks):
            if i < num_codebooks:
                # Active codebooks for evaluation
                cb.trainable = False
                cb.set_active_n_embed(num_embeddings)
                cb.set_frozen_n_embed(0)
            else:
                # Inactive codebooks
                cb.trainable = False
                cb.set_active_n_embed(0)
                cb.set_frozen_n_embed(0)

    def sync_all_ema_weights(self):
        """
        Synchronizes EMA weights to main weights for all codebooks.
        
        Note: This is usually NOT needed during normal training, as VQEmbedding.forward()
        automatically calls _update_embedding() to sync weights after each batch.
        
        Use this method only in special cases:
        - Before saving checkpoints if you want to ensure absolutely latest EMA state
        - In distributed training scenarios for manual synchronization across GPUs
        - For debugging or verification purposes
        - When directly manipulating EMA buffers outside of forward pass
        
        For normal progressive training, the automatic updates in forward() are sufficient.
        """
        for cb in self.codebooks:
            if hasattr(cb, 'sync_ema_weights'):
                cb.sync_ema_weights()
    
    def get_codebook_stats(self):
        """
        Returns statistics for all codebooks (useful for debugging/monitoring).
        
        Returns:
            list: List of dictionaries containing stats for each codebook
        """
        stats_list = []
        for i, cb in enumerate(self.codebooks):
            if hasattr(cb, 'get_codebook_stats'):
                stats = cb.get_codebook_stats()
                stats['codebook_idx'] = i
                stats_list.append(stats)
        return stats_list
    
    def print_codebook_status(self):
        """Prints a summary of the current codebook configuration."""
        status_lines = ["Codebook Status:"]
        for i, cb in enumerate(self.codebooks):
            trainable_str = "trainable" if cb.trainable else "frozen"
            status_lines.append(
                f"  CB{i}: {trainable_str} | active=[0:{cb.active_n_embed}] | "
                f"frozen=[0:{cb.frozen_n_embed}] | training={cb.n_trainable} embeds"
            )
        return "\n".join(status_lines)

    def forward(self, x):
        """
        Forward pass: applies residual quantization.
        
        The process iterates through codebooks, each quantizing the residual from
        the previous stage. Trainable codebooks use EMA updates, while frozen
        codebooks only participate in the forward computation.
        
        Args:
            x (Tensor): Input latent features of shape (B, H, W, C)
            
        Returns:
            quants_final (Tensor): Quantized output of shape (B, H, W, C)
            final_vq_loss (Tensor): VQ loss (commitment loss)
            final_codebook_loss (Tensor): Codebook loss
            final_codes (Tensor): Quantization codes of shape (B, h, w, depth)
        """
        x_reshaped = self.to_code_shape(x)
        residual = x_reshaped.detach().clone()
        aggregated_quants = torch.zeros_like(x_reshaped)
        
        code_list = []
        vq_losses = []
        codebook_losses = []

        for codebook in self.codebooks:
            # Skip inactive codebooks
            if codebook.active_n_embed == 0:
                continue

            # Find nearest embedding indices for current residual
            with torch.no_grad():
                codes = codebook.find_nearest_embedding(residual)

            # Get embedding vectors
            # Sanity checks to avoid CUDA device-side asserts (convert to clear Python errors)
            try:
                # ensure indices are integer type
                if not codes.dtype == torch.long:
                    codes = codes.long()

                # check index range against embedding table
                num_embeds = codebook.weight.shape[0]
                if codes.numel() > 0:
                    min_idx = int(codes.min().item())
                    max_idx = int(codes.max().item())
                    if min_idx < 0 or max_idx >= num_embeds:
                        raise IndexError(
                            f"Embedding index out of range for codebook: min={min_idx}, max={max_idx}, num_embeddings={num_embeds}"
                        )

                # verify devices match (indices and weight)
                if codes.device != codebook.weight.device:
                    raise RuntimeError(
                        f"Device mismatch: codes.device={codes.device}, codebook.weight.device={codebook.weight.device}"
                    )

            except Exception:
                # Re-raise with additional context for easier debugging
                raise

            quant = codebook.embed(codes)

            # Update residual and accumulate quantized values
            if codebook.trainable:
                if codebook.use_ema and self.training:
                    # EMA mode: Update codebook via EMA
                    codebook._update_buffers(residual, codes)
                    codebook._update_embedding()
                    
                    # Detach to prevent gradient flow in EMA mode
                    residual = residual - quant.detach()
                    aggregated_quants = aggregated_quants + quant.detach()
                else:
                    # Gradient mode: Compute VQ losses with gradients
                    vq_losses.append(F.mse_loss(residual.detach(), quant))
                    codebook_losses.append(F.mse_loss(quant.detach(), residual))
                    
                    # Update residual and accumulate quantized values (keep gradients)
                    residual = residual.detach() - quant.detach()
                    aggregated_quants = aggregated_quants + quant
            else:
                # Frozen codebook: Detach everything
                residual = residual - quant.detach()
                aggregated_quants = aggregated_quants + quant.detach()

            code_list.append(codes.unsqueeze(-1))

        # Handle edge case: no active codebooks
        if not code_list:
            final_codes = torch.empty(*self.code_shape[:-1], 0, device=x.device, dtype=torch.long)
            zero_loss = torch.tensor(0.0, device=x.device)
            return x, zero_loss, zero_loss, final_codes

        # Concatenate codes from all codebooks
        final_codes = torch.cat(code_list, dim=-1)

        # Compute final losses
        if self.training and vq_losses:
            final_vq_loss = torch.mean(torch.stack(vq_losses))
            final_codebook_loss = torch.mean(torch.stack(codebook_losses))
        else:
            final_vq_loss = torch.tensor(0.0, device=x.device)
            final_codebook_loss = torch.tensor(0.0, device=x.device)

        # Reshape to original latent shape
        quants_final = self.to_latent_shape(aggregated_quants)

        return quants_final, final_vq_loss, final_codebook_loss, final_codes

    @torch.no_grad()
    def embed_code(self, code):
        """
        Embeds quantization codes back to latent space.
        
        Args:
            code (Tensor): Quantization codes
            
        Returns:
            Tensor: Reconstructed latent features
        """
        assert code.shape[1:] == self.code_shape
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        embeds = torch.cat(embeds, dim=-2).sum(-2)
        embeds = self.to_latent_shape(embeds)
        return embeds
    
    @torch.no_grad()
    def embed_code_with_depth(self, code, to_latent_shape=False):
        """
        Embeds codes while preserving depth dimension.
        
        Args:
            code (Tensor): Quantization codes
            to_latent_shape (bool): Whether to reshape to latent shape
            
        Returns:
            Tensor: Embedded features with depth dimension
        """
        assert code.shape[-1] == self.code_shape[-1]
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        if to_latent_shape:
            embeds = [self.to_latent_shape(embed.squeeze(-2)).unsqueeze(-2) for embed in embeds]
        embeds = torch.cat(embeds, dim=-2)
        return embeds, None

    @torch.no_grad()
    def embed_partial_code(self, code, code_idx, decode_type='select'):
        """
        Embeds partial codes up to a specific depth.
        
        Args:
            code (Tensor): Quantization codes
            code_idx (int): Depth index to decode up to
            decode_type (str): 'select' to use only code_idx, 'add' to sum up to code_idx
            
        Returns:
            Tensor: Partially reconstructed features
        """
        assert code.shape[1:] == self.code_shape
        assert code_idx < code.shape[-1]
        B, h, w, _ = code.shape
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        if decode_type == 'select':
            embeds = embeds[code_idx].view(B, h, w, -1)
        elif decode_type == 'add':
            embeds = torch.cat(embeds[:code_idx+1], dim=-2).sum(-2)
        else:
            raise NotImplementedError(f"{decode_type} is not implemented in partial decoding")
        embeds = self.to_latent_shape(embeds)
        return embeds

    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):
        """
        Computes soft (probabilistic) quantization codes.
        
        Args:
            x (Tensor): Input features
            temp (float): Temperature for softmax
            stochastic (bool): Whether to sample stochastically
            
        Returns:
            soft_code (Tensor): Soft assignment probabilities
            code (Tensor): Hard assignment indices
        """
        x = self.to_code_shape(x)
        residual_feature = x.detach().clone()
        soft_code_list = []
        code_list = []
        n_codebooks = self.code_shape[-1]
        for i in range(n_codebooks):
            codebook = self.codebooks[i]
            distances = codebook.compute_distances(residual_feature)
            soft_code = F.softmax(-distances / temp, dim=-1)
            if stochastic:
                soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
                code = torch.multinomial(soft_code_flat, 1)
                code = code.reshape(*soft_code.shape[:-1])
            else:
                code = distances.argmin(dim=-1)
            quants = codebook.embed(code)
            residual_feature -= quants
            code_list.append(code.unsqueeze(-1))
            soft_code_list.append(soft_code.unsqueeze(-2))
        code = torch.cat(code_list, dim=-1)
        soft_code = torch.cat(soft_code_list, dim=-2)
        return soft_code, code


def visualize_rq_training_state(rq_bottleneck, save_path=None, figsize=(15, 3), dpi=100):
    
    n_codebooks = len(rq_bottleneck.codebooks)
    
    # Create figure with single row for usage rate
    fig, axes = plt.subplots(1, n_codebooks, figsize=figsize, dpi=dpi)
    if n_codebooks == 1:
        axes = [axes]
    
    for i, codebook in enumerate(rq_bottleneck.codebooks):
        n_embed = codebook.n_embed
        active_n = codebook.active_n_embed
        frozen_n = codebook.frozen_n_embed
        
        # ============================================================
        # Usage Rate (使用率)
        # ============================================================
        ax = axes[i]
        
        if hasattr(codebook, 'cluster_size_ema'):
            usage = codebook.cluster_size_ema.cpu().numpy()
            
            x_indices = np.arange(n_embed)
            colors = []
            for idx in x_indices:
                if idx < frozen_n:
                    colors.append('royalblue')
                elif idx < active_n:
                    colors.append('limegreen')
                else:
                    colors.append('lightgray')
            
            bars = ax.bar(x_indices, usage, color=colors, alpha=0.7)
            
            # Mark regions
            if frozen_n > 0:
                ax.axvspan(-0.5, frozen_n-0.5, alpha=0.1, color='blue')
            if active_n > frozen_n:
                ax.axvspan(frozen_n-0.5, active_n-0.5, alpha=0.1, color='green')
            if active_n < n_embed:
                ax.axvspan(active_n-0.5, n_embed-0.5, alpha=0.1, color='gray')
            
            # Mark unused codes threshold
            ax.axhline(y=codebook.eps, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label=f'Unused threshold')
            
            ax.set_title(f'Codebook {i}: Embedding Index Usage Rate\n'
                        f'Frozen:[0:{frozen_n}] Trainable:[{frozen_n}:{active_n}] Total:{n_embed}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Embedding Index', fontsize=9)
            ax.set_ylabel('Usage Count (EMA Cluster Size)', fontsize=9)
            ax.set_yscale('log')  # Log scale for better visualization
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(fontsize=8, loc='upper right')
            
            # Add statistics
            n_unused = (usage < codebook.eps).sum()
            frozen_usage = usage[:frozen_n] if frozen_n > 0 else np.array([])
            trainable_usage = usage[frozen_n:active_n] if active_n > frozen_n else np.array([])
            
            stats_text = f'Unused: {n_unused}/{n_embed}\n'
            if len(frozen_usage) > 0:
                stats_text += f'Frozen avg: {frozen_usage.mean():.2e}\n'
            if len(trainable_usage) > 0:
                stats_text += f'Train avg: {trainable_usage.mean():.2e}'
            
            ax.text(0.98, 0.97, stats_text.strip(), transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        else:
            # No usage data available
            ax.text(0.5, 0.5, 'No usage data\n(EMA not enabled)', 
                    ha='center', va='center', transform=ax.transAxes, 
                    fontsize=10, color='gray')
            ax.set_title(f'Codebook {i}: Usage Rate', fontsize=10, fontweight='bold')
            ax.set_xlabel('Embedding Index', fontsize=9)
    
    # Overall title
    trainable_status = "✓ Training" if rq_bottleneck.codebooks[0].trainable else "✗ Frozen"
    mode_str = "EMA" if rq_bottleneck.codebooks[0].use_ema else "Gradient"
    title = f'RQBottleneck Embedding Usage Monitor ({trainable_status} | {mode_str} Mode)'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save or return
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig
