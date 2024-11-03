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

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism. The individual attention heads aren't split into seperate
    modules, because this would reduce the ability to perform efficient parallel computation.
    
    Args:
        d_model (int):                  Dimensionality of the model (Embedding dimension).
        num_heads (int):                Number of attention heads.
        context_window (int):           Size of the context window. Defaults to d_model.
        masked (bool):                  Whether to use causal masked attention or not.
    """
    def __init__( 
            self, 
            d_model: int, 
            num_heads: int, 
            context_window: int=None, 
            masked: bool=False,
        ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.nh         = num_heads
        self.d_k        = d_model // self.nh # Embedding dimension for each head
        self.d_model    = d_model # Why don't we split the heads?
        self.ctx_d      = context_window if context_window is not None else d_model # Context window size
        self.masked     = masked

        # Weight matrices for keys, queryes and values for all heads. Batched to increase efficiency.
        self.attn = nn.Linear(d_model, 3 * d_model)
        
        # Final linear projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Causal mask. Lower triangular matrix with ones on/below the diagonal.
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(self.ctx_d, self.ctx_d)).view(1, 1, self.ctx_d, self.ctx_d) # (1, 1, ctx_d, ctx_d)
        )
        
    def forward( self, x: Tensor ) -> Tensor:
        B, T, C = x.size() # (batch_size, seq_length, d_model)

        # Calculate queries, keys and values for all heads in a batch, and move head dimension to the front
        Q, K, V = self.attn(x).split(self.d_model, dim=2) # (B, T, 3*d_model) -> 3 * (B, T, d_model)
        Q = Q.view(B, T, self.nh, self.d_k).transpose(1, 2) # (B, T, nh, d_k) -> (B, nh, T, d_k)
        K = K.view(B, T, self.nh, self.d_k).transpose(1, 2) # (B, T, nh, d_k) -> (B, nh, T, d_k)
        V = V.view(B, T, self.nh, self.d_k).transpose(1, 2) # (B, T, nh, d_k) -> (B, nh, T, d_k)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.masked:
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        # Store attention maps for hooks
        self.attn_weights = attn

        # Compute the attention output
        y = attn @ V
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        y = self.W_o(y)
        return y
        
class TransformerBlock(nn.Module):
    """
    Transformer block consisting of: 
        1. Multi-Head Attention
        2. Add & Norm
        3. Feed Forward (Multi-Layer Perceptron)
        4. Add & Norm
    """
    def __init__( 
            self,
            num_heads:  int,
            d_model:    int,
            hidden_dim: int,
            masked:     bool=False,
            dropout:    float=0.1, 
        ) -> None:
        """
        Args:
            num_heads (int):                Number of attention heads.
            d_model (int):                  Dimensionality of the model (Embedding dimension).
            hidden_dim (int):               Hidden layer dimensionality (MLP).
            masked (bool, optional):        Whether to use causal masked attention or not. Defaults to False.
            dropout (float, optional):      Dropout rate. Defaults to 0.1.
        """
        super(TransformerBlock, self).__init__()
        self.attn   = MultiHeadAttention(d_model, num_heads, masked=masked)
        self.ln1    = nn.LayerNorm(d_model)
        self.mlp    = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.ln2    = nn.LayerNorm(d_model)
        self.do     = nn.Dropout(dropout)
    
    def forward( self, x: Tensor ) -> Tensor:
        """
        Forward pass through the transformer block. Results from mlp and multi-head attention
        are added to the input tensor to create residual connections.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.do(self.mlp(self.ln2(x)))
        return x
    
class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute positional encoding module using learnable embeddings.
    """
    def __init__( self, d_model: int, dropout: float = 0.1, max_length: int = 5000 ) -> None:
        """
        Args:
            d_model (int):                  Dimension of embeddings.
            dropout (float, optional):      Dropout rate, defaults to 0.1.
            max_length (int, optional):     Max sequence length, defaults to 5000.
        """
        super().__init__()
        self.do = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_length, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            Tensor: The input tensor with positional embeddings added.
        """
        B, T, C = x.size() # (batch_size, seq_length, d_model)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T) # (0, 1, ..., seq_length-1) for each batch
        pe = self.pe(pos)
        x = x + pe
        x = self.do(x)
        return x
    
class PositionalEncoding(nn.Module):
    """
    Positional encoding module. Adds positional encodings to the input embeddings using
    sine and cosine functions as described in the "Attention is All You Need" paper.
    
    Follows tutorial:
        https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    """
    def __init__( self, d_model: int, dropout: float = 0.1, max_length: int = 5000 ) -> None:
        """
        Args:
            d_model (int):                  Dimension of embeddings
            dropout (float, optional):      Dropout rate, defaults to 0.1.
            max_length (int, optional):     Max sequence length, defaults to 5000.
        """
        super().__init__()
        self.do = nn.Dropout(p=dropout)
        pe = torch.zeros(max_length, d_model)

        # Create initial position column (0, 1, 2, ..., max_length)
        k = torch.arange(0, max_length).unsqueeze(1)

        # Calculate normalization term
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # Calculate sine on even indices and cosine on odd indices
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0) # (1, max_length, d_model)

        # Buffer to store positional encodings to make it non-trainable                        
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor (batch_size, seq_length, d_model)
        
        Returns:
            Tensor: x + positional encodings (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
        x = self.do(x)
        return x

class Encoder(nn.Module):
    """
    Transformer encoder. Consists of:
        1. Embedding layer
        2. Positional encoding
        3. Transformer blocks
    
    The input should be tokenized indicies, and the output will be the encoded representation in the embedding space.
    """
    def __init__( 
            self,
            vocab_size:     int,
            d_model:        int,
            num_heads:      int,
            hidden_dim:     int,
            num_blocks:     int,
            dropout:        float=0.1,
            max_pe_length:  int=5000,
            echo_specs:     bool=True
        ) -> None:
        """
        Args:
            vocab_size (int):               Size of the vocabulary.
            d_model (int):                  Dimensionality of the model (Embedding dimension).
            num_heads (int):                Number of attention heads.
            hidden_dim (int):               Hidden layer dimensionality (MLP).
            num_blocks (int):               Number of transformer blocks.
            dropout (float, optional):      Dropout rate. Defaults to 0.1.
            max_pe_length (int, optional):  Max positional encoding length. Defaults to 5000.
            echo_specs (bool, optional):    Whether to print the model specifications. Defaults to True.
        """
        super(Encoder, self).__init__()
        self.vocab_size     = vocab_size
        self.d_model        = d_model
        self.num_heads      = num_heads
        self.hidden_dim     = hidden_dim
        self.num_blocks     = num_blocks
        self.dropout        = dropout
        self.max_pe_length  = max_pe_length
        self.echo_specs     = echo_specs
        
        self.emb            = nn.Embedding(vocab_size, d_model)
        self.pos_enc        = AbsolutePositionalEncoding(d_model, dropout, max_length=max_pe_length)
        self.blocks         = nn.ModuleList([
            TransformerBlock(num_heads, d_model, hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.echo_specs = echo_specs
        if echo_specs: print(self)
        
    def __repr__( self ) -> str:
        """
        Returns a string representation of the model specifications.
        """
        # Calculate total trainable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Build the string
        model_str = f"\n\rEncoder Model Specifications:\n"
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
            ('Positional Encoding:', self.pos_enc),
        ]

        # Add Transformer Blocks
        for i, block in enumerate(self.blocks):
            components.append((f'Transformer Block {i+1}:', block))

        # Calculate and append parameter counts for each component
        for name, module in components:
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            model_str += f"  * {name} {num_params}\n"

        model_str += f"{'='*40}\n"
        return model_str            
    
    def forward( self, x: Tensor ) -> Tensor:
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor. (batch_size, seq_length)

        Returns:
            Tensor: Encoded representation. (batch_size, seq_length, d_model)
        """
        x = self.emb(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    """
    Decoder model. Consists of:
        1. Embedding layer
        2. Positional encoding
        3. Transformer blocks (masked)
    """
    def __init__( 
            self, 
            vocab_size: int, 
            d_model: int, 
            num_heads: int, 
            hidden_dim: int, 
            num_blocks: int, 
            dropout: float=0.1,
            echo_specs: bool=True
        ) -> None:
        """
        Args:
            vocab_size (int):               Size of the vocabulary.
            d_model (int):                  Dimensionality of the model (Embedding dimension).
            num_heads (int):                Number of attention heads.
            hidden_dim (int):               Hidden layer dimensionality (MLP).
            num_blocks (int):               Number of transformer blocks.
            dropout (float, optional):      Dropout rate. Defaults to 0.1.
            echo_spechs (bool, optional):   Whether to print the model specifications. Defaults to True.
        """
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = AbsolutePositionalEncoding(d_model, dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(num_heads=num_heads, d_model=d_model, hidden_dim=hidden_dim, masked=True, dropout=dropout) for _ in range(num_blocks)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        if echo_specs: print(self)
    
    def __repr__( self ) -> str:
        """
        Returns a string representation of the model specifications.
        """
        # Calculate total trainable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Build the string
        model_str = f"\n\rDecoder Model Specifications:\n"
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
            ('Positional Encoding:', self.pos_enc),
            ('Linear Head:        ', self.lm_head),
            ('Layer Norm:         ', self.ln_f),
        ]

        # Add Transformer Blocks
        for i, block in enumerate(self.blocks):
            components.append((f'Transformer Block {i+1}:', block))

        # Calculate and append parameter counts for each component
        for name, module in components:
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            model_str += f"  * {name} {num_params}\n"

        model_str += f"{'='*40}\n"
        return model_str
    
    def forward( self, x: Tensor, targets: Tensor=None ) -> Tensor:
        """
        Forward pass through the decoder.

        Args:
            x (Tensor):                     Input tensor. (batch_size, seq_length)
            targets (Tensor, optional):     Target tensor. Defaults to None.

        Returns:
            x (Tensor):                     Decoded representation. (batch_size, seq_length, d_model)
            loss (Tensor):                  Loss value. Defaults to None.
            
        """
        x = self.emb(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return x, loss
        
class CLSModel(nn.Module):
    """
    Classifier model. Consists of:
        1. Encoder
        2. Classification layer
    """
    def __init__( self, encoder: Encoder, n_hidden: int, num_classes: int, echo_specs: bool=True ) -> None:
        """
        Args:
            encoder (Encoder):              Encoder model.
            n_hidden (int):                 Hidden layer dimensionality.
            num_classes (int):              Number of classes.
            echo_specs (bool, optional):    Whether to print the model specifications. Defaults to True.
        """
        super(CLSModel, self).__init__()
        self.encoder    = encoder
        self.num_heads  = encoder.num_heads
        self.ff         = nn.Linear(encoder.d_model, n_hidden)
        self.cls        = nn.Linear(n_hidden, num_classes)
        if echo_specs: print(self)
        
    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_str = f"\n\rCLSModel Specifications:\n"
        model_str += f"{'='*40}\n"
        model_str += f"Total Parameters: {total_params}\n"
        model_str += f"{'='*40}\n"
        model_str += f"Trainable Parameters per Component:\n"
        model_str += f"  * Encoder Parameters: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}\n"
        model_str += f"  * Classifier Parameters: {total_params - sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}\n"
        model_str += f"{'='*40}\n"
        return model_str
    
    def forward( self, x: Tensor ) -> Tensor:
        """
        Forward pass through the classifier. It will average over the sequence dimention on the output
        of the encoder to get a single semantic representation of the input sequence, and then classify it.

        Args:
            x (Tensor): Input tensor. (batch_size, seq_length)

        Returns:
            Tensor: Output tensor. (batch_size, num_classes)
        """
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = F.relu(self.ff(x))
        x = self.cls(x)
        return x
    