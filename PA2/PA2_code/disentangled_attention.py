#########################################################
## CSE 256 - Statistical Natural Language Processing   ##
## Transformer assignment (PA2)                        ##
## --------------------------------------------------- ##
## Author:   Mathias Otnes                             ##
## Date:     2024-10-22                                ##
#########################################################

#######################
## Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

#######################
## Implementation

class DisentangledSelfAttention(nn.Module):
    """
    Disentangled Self-Attention mechanism with relative positional embeddings.
    This implementation follows the same functionality as Microsoft's version:
     - https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/disentangled_attention.py

    Args:
        d_model (int):              Dimensionality of the model (Embedding dimension).
        num_heads (int):            Number of attention heads.
        max_position_embeddings     (int): Maximum sequence length for positional embeddings.
        position_buckets (int):     Number of position buckets for relative positions.
        dropout (float):            Dropout rate.
        attention_dropout (float):  Dropout rate for attention probabilities.
        pos_att_type (str):         Type of positional attention ('c2p', 'p2c', 'p2p', etc.).
        share_att_key (bool):       Whether to share attention keys.
        masked (bool):              Whether to use causal masked attention or not.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_position_embeddings: int = 512,
        position_buckets: int = -1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pos_att_type: str = 'c2p|p2c',
        share_att_key: bool = False,
        masked: bool = False,
    ) -> None:
        super(DisentangledSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model                = d_model
        self.num_heads              = num_heads
        self.d_k                    = d_model // num_heads  # Dimension per head
        self.all_head_size          = self.num_heads * self.d_k
        self.dropout                = dropout
        self.attention_dropout      = attention_dropout
        self.pos_att_type           = pos_att_type.split('|')
        self.share_att_key          = share_att_key
        self.masked                 = masked

        # Linear projections for queries, keys, and values
        self.query_proj             = nn.Linear(d_model, self.all_head_size)
        self.key_proj               = nn.Linear(d_model, self.all_head_size)
        self.value_proj             = nn.Linear(d_model, self.all_head_size)
        self.out_proj               = nn.Linear(d_model, d_model)

        # Relative positional embeddings
        self.relative_attention     = True
        self.position_buckets       = position_buckets
        self.max_relative_positions = max_position_embeddings

        # Positional embedding size
        self.pos_ebd_size = self.max_relative_positions
        if self.position_buckets > 0:
            self.pos_ebd_size = self.position_buckets

        # Positional embeddings
        self.pos_dropout = nn.Dropout(dropout)
        self.rel_embeddings = nn.Embedding(self.pos_ebd_size * 2, self.d_model)

        # Additional projections if attention keys are not shared
        if not self.share_att_key:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_key_proj = nn.Linear(d_model, self.all_head_size)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_query_proj = nn.Linear(d_model, self.all_head_size)

        # Causal mask for masked attention
        if self.masked:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(self.max_relative_positions, self.max_relative_positions))
                    .view(1, 1, self.max_relative_positions, self.max_relative_positions)
            )

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """
        Transpose the input tensor for multi-head attention computation.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, all_head_size).

        Returns:
            Tensor: Transposed tensor of shape (batch_size * num_heads, seq_length, head_size).
        """
        new_x_shape = x.size()[:-1] + (self.num_heads, self.d_k)
        x = x.view(*new_x_shape)  # (batch_size, seq_length, num_heads, head_size)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_size)
        return x.reshape(-1, x.size(2), self.d_k)  # (batch_size * num_heads, seq_length, head_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for disentangled self-attention.

        Args:
            x (Tensor):                         Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        B, T, C = x.size()

        # Linear projections
        Q = self.query_proj(x)  # (B, T, all_head_size)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Transpose for scores
        Q = self.transpose_for_scores(Q)  # (B * nh, T, d_k)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        # Build relative positions
        relative_pos = self.build_relative_position(T, T, device=x.device)
        rel_embeddings = self.pos_dropout(self.rel_embeddings.weight)

        # Scaled dot-product attention with disentangled biases
        scale_factor = self.compute_scale_factor()
        scale = 1 / math.sqrt(self.d_k * scale_factor)
        attention_scores = torch.bmm(Q, K.transpose(-1, -2)) * scale

        # Add disentangled attention biases
        rel_att = self.disentangled_attention_bias(Q, K, relative_pos, rel_embeddings, scale_factor)
        attention_scores += rel_att

        # Apply causal mask if masked
        if self.masked:
            attention_scores = attention_scores.view(B, self.num_heads, T, T)
            attention_scores = attention_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            attention_scores = attention_scores.view(-1, T, T)

        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        self.attn_weights = attention_probs.clone().detach()
        attention_probs = F.dropout(attention_probs, p=self.attention_dropout, training=self.training)

        # Compute the attention output
        context_layer = torch.bmm(attention_probs, V)  # (B * num_heads, T, d_k)
        context_layer = context_layer.view(B, self.num_heads, T, self.d_k)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (B, T, num_heads, d_k)
        context_layer = context_layer.view(B, T, -1)  # (B, T, d_model)

        # Final linear projection
        output = self.out_proj(context_layer)

        return output

    def compute_scale_factor(self) -> int:
        """
        Compute the scale factor based on the positional attention types.

        Returns:
            int: Scale factor.
        """
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        return scale_factor

    def build_relative_position(self, qlen: int, klen: int, device: torch.device) -> Tensor:
        """
        Build relative position matrix for the input sequences.

        Args:
            qlen (int): Query sequence length.
            klen (int): Key sequence length.
            device (torch.device): Device to place the tensor.

        Returns:
            Tensor: Relative position tensor of shape (1, qlen, klen).
        """
        context_position = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        if self.position_buckets > 0:
            relative_position = self.relative_position_bucket(
                relative_position,
                num_buckets=self.position_buckets,
                max_distance=self.max_relative_positions
            )
        else:
            relative_position = torch.clamp(
                relative_position + self.max_relative_positions,
                0,
                2 * self.max_relative_positions - 1
            )
        return relative_position[None, :, :]  # (1, qlen, klen)

    def relative_position_bucket(self, relative_positions: Tensor, num_buckets: int, max_distance: int) -> Tensor:
        """
        Translate relative positions to relative position buckets.

        Args:
            relative_positions (Tensor): Relative positions.
            num_buckets (int): Number of buckets.
            max_distance (int): Maximum distance.

        Returns:
            Tensor: Relative position buckets.
        """
        relative_buckets = 0
        n = -relative_positions
        n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6) /
            math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        relative_buckets = torch.where(is_small, n, val_if_large)
        return relative_buckets

    def disentangled_attention_bias(
        self,
        query_layer: Tensor,
        key_layer: Tensor,
        relative_pos: Tensor,
        rel_embeddings: Tensor,
        scale_factor: int
    ) -> Tensor:
        """
        Compute disentangled attention biases.

        Args:
            query_layer (Tensor):       Query tensor (B * num_heads, T, d_k).
            key_layer (Tensor):         Key tensor (B * num_heads, T, d_k).
            relative_pos (Tensor):      Relative position tensor (1, T, T).
            rel_embeddings (Tensor):    Relative positional embeddings.
            scale_factor (int):         Scale factor for attention scores.

        Returns:
            Tensor: Attention biases tensor (B * num_heads, T, T).
        """
        B_num_heads, T, _ = query_layer.size()
        att_span = self.pos_ebd_size

        # Adjust relative positions
        relative_pos = relative_pos.to(query_layer.device)
        rel_pos = torch.clamp(
            relative_pos + att_span,
            0,
            att_span * 2 - 1
        ).long()  # (1, T, T)

        # Content-to-Position attention
        if 'c2p' in self.pos_att_type:
            pos_key = self.pos_key_proj(rel_embeddings) if not self.share_att_key else self.key_proj(rel_embeddings)
            pos_key = pos_key.view(-1, self.num_heads, self.d_k).permute(1, 0, 2)  # (num_heads, 2 * att_span, d_k)
            pos_key = pos_key.repeat(B_num_heads // self.num_heads, 1, 1)  # (B_num_heads, 2 * att_span, d_k)
            pos_key = pos_key.reshape(B_num_heads, 2 * att_span, self.d_k)
            scale = 1 / math.sqrt(self.d_k * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key.transpose(-1, -2)) * scale  # (B_num_heads, T, 2 * att_span)
            c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=rel_pos.expand(B_num_heads, T, T)
            )  # (B_num_heads, T, T)
        else:
            c2p_att = torch.zeros_like(query_layer @ key_layer.transpose(-1, -2))

        # Position-to-Content attention
        if 'p2c' in self.pos_att_type:
            pos_query = self.pos_query_proj(rel_embeddings) if not self.share_att_key else self.query_proj(rel_embeddings)
            pos_query = pos_query.view(-1, self.num_heads, self.d_k).permute(1, 0, 2)  # (num_heads, 2 * att_span, d_k)
            pos_query = pos_query.repeat(B_num_heads // self.num_heads, 1, 1)  # (B_num_heads, 2 * att_span, d_k)
            pos_query = pos_query.reshape(B_num_heads, 2 * att_span, self.d_k)
            scale = 1 / math.sqrt(self.d_k * scale_factor)
            p2c_att = torch.bmm(pos_query, key_layer.transpose(-1, -2)) * scale  # (B_num_heads, 2 * att_span, T)
            p2c_att = torch.gather(
                p2c_att,
                dim=-2,
                index=rel_pos.transpose(-1, -2).expand(B_num_heads, T, T)
            ).transpose(-1, -2)  # (B_num_heads, T, T)
        else:
            p2c_att = torch.zeros_like(query_layer @ key_layer.transpose(-1, -2))

        # Sum up the biases
        att_bias = c2p_att + p2c_att

        return att_bias

class DisentangledTransformerBlock(nn.Module):
    """
    Transformer block with disentangled self-attention mechanism.
    Consists of:
        1. Disentangled Self-Attention
        2. Add & Norm
        3. Feed Forward (Multi-Layer Perceptron)
        4. Add & Norm
    """
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        hidden_dim: int,
        masked: bool = False,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 512,
        position_buckets: int = -1,
        pos_att_type: str = 'c2p|p2c',
        share_att_key: bool = False,
    ) -> None:
        """
        Args:
            num_heads (int):                            Number of attention heads.
            d_model (int):                              Dimensionality of the model (Embedding dimension).
            hidden_dim (int):                           Hidden dimension of the feed-forward layer.
            masked (bool, optional):                    Whether to use masked attention. Defaults to False.
            dropout (float, optional):                  Dropout rate. Defaults to 0.1.
            attention_dropout (float, optional):        Dropout rate for attention probabilities. Defaults to 0.1.
            max_position_embeddings (int, optional):    Maximum sequence length for positional embeddings. Defaults to 512.
            position_buckets (int, optional):           Number of position buckets for relative positions. Defaults to -1.
            pos_att_type (str, optional):               Type of positional attention ('c2p', 'p2c', 'p2p', etc.). Defaults to 'c2p|p2c'.
            share_att_key (bool, optional):             Whether to share attention keys. Defaults to False.
        """
        super(DisentangledTransformerBlock, self).__init__()
        self.attn = DisentangledSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            position_buckets=position_buckets,
            dropout=dropout,
            attention_dropout=attention_dropout,
            pos_att_type=pos_att_type,
            share_att_key=share_att_key,
            masked=masked,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.do(self.mlp(self.ln2(x)))
        return x

class DisentangledDecoder(nn.Module):
    """
    Disentangled Transformer decoder.
    Consists of:
        1. Embedding layer
        2. Disentangled Transformer blocks (with masked attention)
        3. Layer Norm
        4. Linear Head (for language modeling)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 512,
        position_buckets: int = -1,
        pos_att_type: str = 'c2p|p2c',
        share_att_key: bool = False,
        echo_specs: bool = True,
    ) -> None:
        """
        Args:
            vocab_size (int):                           Size of the vocabulary.
            d_model (int):                              Dimensionality of the model (Embedding dimension).
            num_heads (int):                            Number of attention heads.
            hidden_dim (int):                           Hidden layer dimensionality (MLP).
            num_blocks (int):                           Number of transformer blocks.
            dropout (float, optional):                  Dropout rate. Defaults to 0.1.
            attention_dropout (float, optional):        Dropout rate for attention probabilities. Defaults to 0.1.
            max_position_embeddings (int, optional):    Maximum sequence length for positional embeddings. Defaults to 512.
            position_buckets (int, optional):           Number of position buckets for relative positions. Defaults to -1.
            pos_att_type (str, optional):               Type of positional attention ('c2p', 'p2c', 'p2p', etc.). Defaults to 'c2p|p2c'.
            share_att_key (bool, optional):             Whether to share attention keys. Defaults to False.
            echo_specs (bool, optional):                Whether to print the model specifications. Defaults to True.
        """
        super(DisentangledDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout

        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DisentangledTransformerBlock(
                num_heads=num_heads,
                d_model=d_model,
                hidden_dim=hidden_dim,
                masked=True,
                dropout=dropout,
                attention_dropout=attention_dropout,
                max_position_embeddings=max_position_embeddings,
                position_buckets=position_buckets,
                pos_att_type=pos_att_type,
                share_att_key=share_att_key,
            ) for _ in range(num_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        if echo_specs:
            print(self)

    def __repr__(self) -> str:
        """
        Returns a string representation of the model specifications.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        model_str = f"\n\rDisentangled Decoder Model Specifications:\n"
        model_str += f"{'='*40}\n"
        model_str += f"Vocabulary Size:          {self.vocab_size}\n"
        model_str += f"Embedding Dimension:      {self.d_model}\n"
        model_str += f"Number of Heads:          {self.num_heads}\n"
        model_str += f"Number of Blocks:         {self.num_blocks}\n"
        model_str += f"Hidden Dimension (MLP):   {self.hidden_dim}\n"
        model_str += f"Dropout Rate:             {self.dropout}\n"
        model_str += f"Total Parameters:         {total_params}\n"
        model_str += f"{'='*40}\n"
        model_str += f"Trainable Parameters per Component:\n"

        # Components and their parameter counts
        components = [
            ('Embedding Layer:    ', self.emb),
            ('Linear Head:        ', self.lm_head),
            ('Layer Norm:         ', self.ln_f),
        ]

        # Add Transformer Blocks
        for i, block in enumerate(self.blocks):
            components.append((f'Disentangled Transformer Block {i+1}:', block))

        # Calculate and append parameter counts for each component
        for name, module in components:
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            model_str += f"  * {name} {num_params}\n"

        model_str += f"{'='*40}\n"
        return model_str

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        """
        Forward pass through the disentangled decoder.

        Args:
            x (Tensor):                     Input tensor. (batch_size, seq_length)
            targets (Tensor, optional):     Target tensor. Defaults to None.

        Returns:
            x (Tensor):                     Decoded representation. (batch_size, seq_length, d_model)
            loss (Tensor):                  Loss value. Defaults to None.
        """
        x = self.emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return x, loss
