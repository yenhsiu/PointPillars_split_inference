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
Vector Quantization (VQ) and Residual Quantization (RQ) Implementation
"""

from typing import Iterable
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class VQEmbedding(nn.Embedding):

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
        self._use_ema = ema # 直接將 ema 參數設為一個可變的屬性
        
        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        self.weight.requires_grad = (not self.use_ema) and self._trainable

    @property
    def use_ema(self):
        return self._use_ema

    @use_ema.setter
    def use_ema(self, value):
        """
        【核心修正】簡化 setter 邏輯。
        設定 use_ema 後，立即根據新狀態更新 requires_grad。
        移除了對 self.trainable 的遞迴呼叫。
        """
        self._use_ema = value
        # 狀態改變後，立即更新梯度需求。
        self.weight.requires_grad = (not self._use_ema) and self._trainable

    def reset_usage_tracking(self):
        self.register_buffer('cluster_size', torch.zeros(self.active_n_embed))

    def set_frozen_n_embed(self, n):
        self.frozen_n_embed = n

    def set_active_n_embed(self, n):
        self.active_n_embed = n

    @torch.no_grad()
    def compute_distances(self, inputs):
        # 如果沒有 active embedding，返回極大的距離
        if self.active_n_embed == 0:
            return torch.full((*inputs.shape[:-1], 1), float('inf'), device=inputs.device)
            
        codebook_t = self.weight[:self.active_n_embed, :].t()
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim
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
        # 如果沒有 active embedding，返回全零的索引
        if self.active_n_embed == 0:
            return torch.zeros(*inputs.shape[:-1], dtype=torch.long, device=inputs.device)
            
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)
        embed_idxs = embed_idxs.clamp(max=self.active_n_embed - 1)
        return embed_idxs

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        if not self._trainable:
            return

        n_active = self.active_n_embed
        embed_dim = self.weight.shape[-1]

        vectors_flat = vectors.reshape(-1, embed_dim)
        idxs_flat = idxs.reshape(-1)
        
        # Clamp indices to be safe
        clamped_idxs = idxs_flat.clamp(max=n_active - 1)
        one_hot = F.one_hot(clamped_idxs, num_classes=n_active).float()
        cluster_size = one_hot.sum(dim=0)

        # 只更新未 frozen 的 embedding
        update_start = self.frozen_n_embed
        vectors_sum_per_cluster = one_hot.t() @ vectors_flat
        self.cluster_size_ema[update_start:n_active].mul_(self.decay).add_(cluster_size[update_start:], alpha=1 - self.decay)
        self.embed_ema[update_start:n_active].mul_(self.decay).add_(vectors_sum_per_cluster[update_start:], alpha=1 - self.decay)
        
        if self.restart_unused_codes:
            unused_indices = torch.where(self.cluster_size_ema[update_start:n_active] < self.eps)[0] + update_start
            n_unused = unused_indices.shape[0]
            if n_unused > 0:
                n_vectors_flat = vectors_flat.shape[0]
                rand_indices = torch.randperm(n_vectors_flat, device=vectors.device)
                if n_vectors_flat < n_unused:
                     n_repeats = (n_unused + n_vectors_flat - 1) // n_vectors_flat
                     rand_indices = rand_indices.repeat(n_repeats)
                random_vectors = vectors_flat[rand_indices[:n_unused]]
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                self.embed_ema[unused_indices] = random_vectors
                self.cluster_size_ema[unused_indices] = torch.ones_like(self.cluster_size_ema[unused_indices]) * (1.0 / world_size)

    @torch.no_grad()
    def _update_embedding(self):
        if not self._trainable:
            return
        n_active = self.active_n_embed
        update_start = self.frozen_n_embed
        n = self.cluster_size_ema[update_start:n_active].sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema[update_start:n_active] + self.eps) / (n + (n_active - update_start) * self.eps)
        )
        # 加入 clamp 避免過小值
        normalized_cluster_size = normalized_cluster_size.clamp(min=self.eps)
        self.weight.data[update_start:n_active, :] = self.embed_ema[update_start:n_active] / normalized_cluster_size.reshape(-1, 1)

    @torch.no_grad()
    def sync_ema_weights(self):
        """
        手動將 EMA buffer 的權重同步到模型的 'weight' 中。
        這在從 EMA 預熱切換到評估或梯度訓練時至關重要。
        
        【關鍵修正】確保只同步有效的 embedding 範圍，避免覆蓋未訓練的部分
        """
        if not self.ema or not hasattr(self, 'embed_ema'):
            return

        # 如果沒有 active embedding，跳過同步
        if self.active_n_embed == 0:
            # print(f"Skipping sync for codebook with active_n_embed=0")
            return

        n_embed = self.active_n_embed
        update_start = self.frozen_n_embed
        
        # 確保範圍有效
        if update_start >= n_embed:
            # print(f"Skipping sync: update_start({update_start}) >= n_embed({n_embed})")
            return
        
        # 處理分母為零的情況
        cluster_size = self.cluster_size_ema[update_start:n_embed]
        n = cluster_size.sum()
        
        # 如果沒有聚類數據，跳過同步
        if n == 0:
            # print(f"Skipping sync: no cluster data for range [{update_start}:{n_embed}]")
            return
            
        normalized_cluster_size = (
            n * (cluster_size + self.eps) / (n + (n_embed - update_start) * self.eps)
        )
        
        # 增加 clamp 防止正規化後的值過小，導致除法不穩定
        normalized_cluster_size = normalized_cluster_size.clamp(min=self.eps)

        # 獲取 EMA 更新後的 embedding
        ema_weights = self.embed_ema[update_start:n_embed] / normalized_cluster_size.unsqueeze(-1)
        
        # 更新主權重張量
        self.weight.data[update_start:n_embed, :] = ema_weights

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        # 【修正】按照官方邏輯，先判斷是否更新 buffers
        if self.training and self._use_ema and self._trainable:
            self._update_buffers(inputs, embed_idxs)
        
        embeds = self.embed(embed_idxs)
        
        # 【修正】按照官方邏輯，在 embed 之後更新 embedding
        if self.training and self._use_ema and self._trainable:
            self._update_embedding()
            
        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """
    Residual Quantizer Bottleneck Module
    """

    def __init__(self,
                 latent_shape,
                 code_shape,
                 n_embed,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 commitment_loss='cumsum',
                 codebook_num=None,
                 ema=True,  # 新增参数
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
        self.codebook_num = codebook_num
        self.shared_codebook = shared_codebook
        
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, (list, tuple)) else [n_embed] * self.code_shape[-1]
        self.decay = decay if isinstance(decay, (list, tuple)) else [decay] * self.code_shape[-1]
        self.ema = ema  # 保存 ema 参数
        assert len(self.n_embed) == self.code_shape[-1]
        assert len(self.decay) == self.code_shape[-1]

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0], 
                                    embed_dim, 
                                    decay=self.decay[0], 
                                    restart_unused_codes=restart_unused_codes,
                                    ema=self.ema  # 使用传入的参数
                                    )
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.code_shape[-1])])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx], 
                                     embed_dim, 
                                     decay=self.decay[idx], 
                                     restart_unused_codes=restart_unused_codes,
                                     ema=self.ema  # 使用传入的参数
                                     ) for idx in range(self.code_shape[-1])]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss
        
    def set_ema_mode(self, ema_mode):
        """切換所有 codebook 的 EMA 模式"""
        for cb in self.codebooks:
            cb.use_ema = ema_mode

    def to_code_shape(self, x):
        (B, H, W, D) = x.shape
        (rH, rW, _) = self.shape_divisor
        x = x.reshape(B, H//rH, rH, W//rW, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H//rH, W//rW, -1)
        return x

    def to_latent_shape(self, x):
        (B, h, w, _) = x.shape
        (_, _, D) = self.latent_shape
        (rH, rW, _) = self.shape_divisor
        x = x.reshape(B, h, w, rH, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h*rH, w*rW, D)
        return x

    def set_training_stage(self, active_codebook_idx, active_embed_size, full_embed_size, prev_embed_size=0):
        """
        設定模型進行漸進式訓練的特定階段。

        Args:
            active_codebook_idx (int): 當前正在訓練的 codebook 的索引。
            active_embed_size (int): 當前 codebook 使用的 embedding 數量。
            full_embed_size (int): 一個 codebook 訓練完成後的 embedding 總數。
            prev_embed_size (int): 上一個 embedding 階段的大小，用於設定 frozen 邊界。
        """
        for i, cb in enumerate(self.codebooks):
            if i == active_codebook_idx:
                # 當前正在訓練的 codebook
                cb.trainable = True
                cb.set_active_n_embed(active_embed_size)
                cb.set_frozen_n_embed(prev_embed_size)
            elif i < active_codebook_idx:
                # 已經訓練完成的 codebook：設為不可訓練並完全凍結
                cb.trainable = False
                cb.set_active_n_embed(full_embed_size)
                cb.set_frozen_n_embed(full_embed_size)
            else:
                # 尚未開始訓練的 codebook：設為不可訓練並關閉
                cb.trainable = False
                cb.set_active_n_embed(0)
                cb.set_frozen_n_embed(0)

    def set_evaluation_stage(self, num_codebooks, num_embeddings):
        """為評估設定 codebook 狀態。"""
        for i, cb in enumerate(self.codebooks):
            if i < num_codebooks:
                # 使用的 codebook
                cb.trainable = False
                cb.set_active_n_embed(num_embeddings)
                cb.set_frozen_n_embed(0)
            else:
                # 不使用的 codebook
                cb.trainable = False
                cb.set_active_n_embed(0)
                cb.set_frozen_n_embed(0)

    def sync_all_ema_weights(self):
        """遍歷所有 codebook 並同步它們的 EMA 權重。"""
        # print("--- Syncing all EMA codebook weights ---")
        for cb in self.codebooks:
            if hasattr(cb, 'sync_ema_weights'):
                cb.sync_ema_weights()

    def forward(self, x):
        """
        Residual Quantization forward with progressive/freeze support.
        只訓練 codebooks，不需要 STE
        """
        x_reshaped = self.to_code_shape(x)
        residual = x_reshaped.detach().clone()
        aggregated_quants = torch.zeros_like(x_reshaped)
        
        code_list = []
        vq_losses = []
        codebook_losses = []

        for codebook in self.codebooks:
            # 完全跳過未啟動的 codebook，不進行任何計算
            if codebook.active_n_embed == 0:
                continue

            with torch.no_grad():
                codes = codebook.find_nearest_embedding(residual)

            quant = codebook.embed(codes)

            # 3. progressive/freeze 控制
            if codebook.trainable:
                if codebook.use_ema and self.training:
                    # EMA 模式：更新 buffers 和 embeddings，但仍計算損失進行監控
                    codebook._update_buffers(residual, codes)
                    codebook._update_embedding()
                    # 計算損失（但不用於反向傳播，因為已經detach）
                    vq_losses.append(F.mse_loss(residual, quant.detach()))
                    codebook_losses.append(F.mse_loss(quant.detach(), residual))
                    residual = residual - quant.detach()
                    aggregated_quants = aggregated_quants + quant.detach()
                else:
                    # 梯度模式：計算 VQ 損失，保持梯度用於 codebook 訓練
                    vq_losses.append(F.mse_loss(residual, quant))
                    codebook_losses.append(F.mse_loss(quant.detach(), residual))
                    # 更新殘差（detach 避免累積計算圖）和累積量化結果（保持梯度）
                    residual = residual.detach() - quant.detach()
                    aggregated_quants = aggregated_quants + quant
            else:
                # 凍結的 codebook：參與計算但不參與訓練
                residual = residual - quant.detach()
                aggregated_quants = aggregated_quants + quant.detach()

            code_list.append(codes.unsqueeze(-1))

        # 處理沒有任何啟動 codebook 的情況
        if not code_list:
            final_codes = torch.empty(*self.code_shape[:-1], 0, device=x.device, dtype=torch.long)
            return x, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device), final_codes

        final_codes = torch.cat(code_list, dim=-1)

        # 損失計算 - 無論EMA還是梯度模式都計算（用於監控）
        if vq_losses:
            final_vq_loss = torch.mean(torch.stack(vq_losses))
            final_codebook_loss = torch.mean(torch.stack(codebook_losses))
        else:
            final_vq_loss = torch.tensor(0.0, device=x.device)
            final_codebook_loss = torch.tensor(0.0, device=x.device)

        # 最終輸出：直接使用累積的量化結果，不使用 STE
        quants_final = self.to_latent_shape(aggregated_quants)
        # 移除 STE：quants_final = x + (quants_final - x).detach()

        return quants_final, final_vq_loss, final_codebook_loss, final_codes

    @torch.no_grad()
    def embed_code(self, code):
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