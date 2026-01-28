import torch
from einops import einsum, rearrange, repeat
import math


### Define basic linear layer

class Linear(torch.nn.Module):
    def __init__(self, d_in : int, d_out : int, device = None, dtype = None):
        super().__init__()

        # declare weight tensor
        weights = torch.empty(d_out, d_in, device=device, dtype=dtype)

        # compute the desired standard deviation
        sigma = math.sqrt(2 / (d_in + d_out))

        # initialize weights to normal distribution, clip weights to [-3 * sigma, 3 * sigma]
        weights = torch.nn.init.trunc_normal_(weights, 0, sigma, -3, 3)

        # turn weights into parameter
        self.weights = torch.nn.Parameter(weights)
    
    def set_weights(self, weights):
        self.load_state_dict({ "weights": weights })

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
    

### Define non-attention blocks (embedding, head, feedforward, norm)

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size : int, d_model : int, device=None, dtype=None):
        super().__init__()

        # declare weight tensor
        weights = torch.empty(vocab_size, d_model, device=device, dtype=dtype)

        # initialize weights to standard normal distribution, clip weights to [-3, 3]
        weights = torch.nn.init.trunc_normal_(weights, 0, 1, -3, 3)

        # turn weights into parameter
        self.weights = torch.nn.Parameter(weights)

    def set_weights(self, weights):
        self.load_state_dict({ "weights": weights })

    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class LMHead(torch.nn.Module):
    def __init__(self, vocab_size : int, d_model : int, device=None, dtype=None):
        super().__init__()

        # create layers
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.unembedding = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def set_weights(self, norm_weights : torch.Tensor, unembedding_weights : torch.Tensor):
        self.norm.set_weights(norm_weights)
        self.unembedding.set_weights(unembedding_weights)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x)
        x = self.unembedding.forward(x)

        return x

def silu(x : torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class FeedForward(torch.nn.Module):
    def __init__(self, d_model : int, d_ff : int | None = None, device = None, dtype = None):
        super().__init__()

        self.d_model = d_model

        # compute feedforward dimension, setting to 8/3 of d_model rounded down to a multiple of 64
        if d_ff is None:
            d_ff = 64 * int((8/3 * d_model) / 64)
        self.d_ff = d_ff

        # declare two linear layers and intermediate gate layer
        self.first_layer = Linear(d_model, d_ff, device, dtype)
        self.gate_layer = Linear(d_model, d_ff, device, dtype)
        self.second_layer = Linear(d_ff, d_model, device, dtype)

    def set_weights(self, first_weights : torch.Tensor, gate_weights : torch.Tensor, second_weights : torch.Tensor):
        self.first_layer.set_weights(first_weights)
        self.gate_layer.set_weights(gate_weights)
        self.second_layer.set_weights(second_weights)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # compute the output of the first layer
        first_layer_output = self.first_layer.forward(x)

        # apply SiLU to the first layer output
        silu_output = silu(first_layer_output)

        # gate the first layer output with the gate layer
        gate_output = silu_output * self.gate_layer.forward(x)

        # pass the gate through the second layer
        second_layer_output = self.second_layer.forward(gate_output)

        return second_layer_output
    
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model : int, epsilon : float = 1e-5, device = None, dtype = None):
        super().__init__()

        self.d_model = d_model
        self.epsilon = epsilon

        # initialize gain values to all be 1 at first
        self.weights = torch.nn.Parameter(torch.ones(d_model))

    def set_weights(self, weights):
        self.load_state_dict({ "weights": weights })

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # upcast input to float32
        x_original = x
        x = x_original.to(torch.float32)

        # get the sum of squares within each token
        squares = torch.square(x)
        sum_of_squares = einsum(squares, "... d_model -> ...")

        # get RMS values and broadcast them to get denominators
        rms = torch.sqrt(sum_of_squares / self.d_model + self.epsilon)
        denominators = repeat(rms, "... -> ... d_model", d_model=self.d_model)

        # divide by denominators and multiply by gains
        x = x / denominators * self.weights

        # downcast
        return x.to(x_original.dtype)


### Define attention block

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.d_k = d_k

        # compute angles for each matrix
        seq_lens = torch.arange(max_seq_len)
        ks = torch.arange(d_k//2)

        numerators = rearrange(seq_lens, "max_seq_len -> max_seq_len 1")
        denominators = rearrange(torch.pow(theta, 2 * ks / d_k), "d_k_split -> 1 d_k_split")

        thetas = numerators / denominators

        # compute cosine and sine values
        cos_vals = torch.cos(thetas)
        sin_vals = torch.sin(thetas)

        # get all rotation matrices
        rotation_matrices = torch.stack([
            torch.stack([cos_vals, -sin_vals], dim=-1),
            torch.stack([sin_vals, cos_vals], dim=-1)
        ], dim=-2)

        # register as static attribute
        self.register_buffer("rotation_matrices", rotation_matrices, persistent=False)

    def forward(self, x : torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # rearrange input into groups of two
        x = rearrange(x, "... seq_len (d_k_split x) -> ... seq_len d_k_split x", x=2)

        # get the correct rotation matrices for each embedding
        embedding_matrices = self.rotation_matrices[token_positions] # type: ignore

        # perform rotations
        x = einsum(embedding_matrices, x, "... seq_len d_k_split y x, ... seq_len d_k_split x -> ... seq_len d_k_split y")

        # merge back
        x = rearrange(x, "... seq_len d_k_split x -> ... seq_len (d_k_split x)")

        return x


def softmax(x : torch.Tensor, dim : int = -1, temperature : float = 1.0) -> torch.Tensor:
    # scale based on temperature value
    scaled = x / temperature

    # move the relevant dimension to the back
    scaled = torch.movedim(scaled, dim, -1)

    # get the max for the relevant dimension and then add extra dimension
    maxes = torch.max(scaled, -1).values
    maxes = rearrange(maxes, "... -> ... 1")

    # shift by the max for the relevant dimension
    shifted = scaled - maxes

    # raise all items to the exponential
    exponentials = torch.exp(shifted)

    # get sums
    sums = torch.sum(exponentials, -1)
    sums = rearrange(sums, "... -> ... 1")

    # get softmaxes
    softmax = exponentials / sums

    # move the relevant dimension back
    return torch.movedim(softmax, -1, dim)

def scaled_dot_product_attention(Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, mask : torch.Tensor | None = None) -> torch.Tensor:
    # get attention scores for all pairs of tokens within each sequence
    attention_matrix = einsum(Q, K, "... seq_len1 d_k, ... seq_len2 d_k -> ... seq_len1 seq_len2")

    # scale all attention scores
    attention_matrix = attention_matrix / math.sqrt(Q.shape[-1])

    # if there is a mask, use it to mask any relevant items
    if mask is not None:
        attention_matrix = attention_matrix.masked_fill(~mask, float("-inf"))

    # run softmax on the attention matrix
    attention_matrix = softmax(attention_matrix)

    # get the correct end scores by summing over the values
    return einsum(attention_matrix, V, "... seq_len1 seq_len2, ... seq_len2 d_v -> ... seq_len1 d_v")

class Attention(torch.nn.Module):
    def __init__(self, d_model : int, num_heads : int, rope : RotaryPositionalEmbedding, device = None, dtype = None):
        super().__init__()

        # initialize dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        # initialize linear layers for query, key, value, and output
        d_k_total = self.d_k * self.num_heads
        d_v_total = self.d_v * self.num_heads
        self.query_layer = Linear(d_model, d_k_total, device, dtype)
        self.key_layer = Linear(d_model, d_k_total, device, dtype)
        self.value_layer = Linear(d_model, d_k_total, device, dtype)
        self.output_layer = Linear(d_v_total, d_model, device, dtype)

        # initialize rope
        self.rope = rope

    def set_weights(self, query_weights : torch.Tensor, key_weights : torch.Tensor, value_weights : torch.Tensor, output_weights : torch.Tensor):
        self.query_layer.set_weights(query_weights)
        self.key_layer.set_weights(key_weights)
        self.value_layer.set_weights(value_weights)
        self.output_layer.set_weights(output_weights)
    
    def multihead_attention(self, x : torch.Tensor) -> torch.Tensor:
        # get Q, K, and V
        Q = self.query_layer.forward(x)
        K = self.key_layer.forward(x)
        V = self.value_layer.forward(x)

        # split Q, K, and V by the head
        Q = rearrange(Q, "... seq_len (heads d_k) -> ... heads seq_len d_k", d_k=self.d_k)
        K = rearrange(K, "... seq_len (heads d_k) -> ... heads seq_len d_k", d_k=self.d_k)
        V = rearrange(V, "... seq_len (heads d_v) -> ... heads seq_len d_v", d_v=self.d_v)

        # construct causal attention mask
        batch_dimensions = Q.shape[:-2]
        sequence_len = Q.shape[-2]
        mask = torch.tril(torch.ones(*batch_dimensions, sequence_len, sequence_len), diagonal=0).to(torch.bool)

        # get attention results and fuse them
        attention_vals = scaled_dot_product_attention(Q, K, V, mask)
        attention_vals = rearrange(attention_vals, "... heads seq_len d_v -> ... seq_len (heads d_v)", d_v=self.d_v)

        # use output layer to recombine attention results
        return self.output_layer.forward(attention_vals)
    
    def multihead_attention_rope(self, x : torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # get Q, K, and V
        Q = self.query_layer.forward(x)
        K = self.key_layer.forward(x)
        V = self.value_layer.forward(x)

        # split Q, K, and V by the head
        Q = rearrange(Q, "... seq_len (heads d_k) -> ... heads seq_len d_k", d_k=self.d_k)
        K = rearrange(K, "... seq_len (heads d_k) -> ... heads seq_len d_k", d_k=self.d_k)
        V = rearrange(V, "... seq_len (heads d_v) -> ... heads seq_len d_v", d_v=self.d_v)

        # construct causal attention mask
        batch_dimensions = Q.shape[:-2]
        sequence_len = Q.shape[-2]
        mask = torch.tril(torch.ones(*batch_dimensions, sequence_len, sequence_len), diagonal=0).to(torch.bool)

        # get token positions
        if token_positions is None:
            token_positions = torch.arange(sequence_len)

        # apply rope to queries and keys
        Q = self.rope.forward(Q, token_positions)
        K = self.rope.forward(K, token_positions)

        # get attention results and fuse them
        attention_vals = scaled_dot_product_attention(Q, K, V, mask)
        attention_vals = rearrange(attention_vals, "... heads seq_len d_v -> ... seq_len (heads d_v)", d_v=self.d_v)

        # use output layer to recombine attention results
        return self.output_layer.forward(attention_vals)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.multihead_attention_rope(x)
    

### Define transformer block

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model : int, num_heads : int, d_ff : int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()

        # declare rope
        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)

        # declare first sublayer
        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = Attention(d_model , num_heads, rope, device=device, dtype=dtype)
        
        # declare second sublayer
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model, d_ff, device=device, dtype=dtype)

    def set_weights(self, rms_norm_1_weights : torch.Tensor, query_weights : torch.Tensor, key_weights : torch.Tensor, value_weights : torch.Tensor, output_weights : torch.Tensor,
                    rms_norm_2_weights : torch.Tensor, ffn_first_weights : torch.Tensor, ffn_gates_weights : torch.Tensor, ffn_second_weights : torch.Tensor):
        self.rms_norm_1.set_weights(rms_norm_1_weights)
        self.attention.set_weights(query_weights, key_weights, value_weights, output_weights)
        self.rms_norm_2.set_weights(rms_norm_2_weights)
        self.ffn.set_weights(ffn_first_weights, ffn_gates_weights, ffn_second_weights)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # compute output of first layer and add to x
        first_layer_output = self.rms_norm_1.forward(x)
        first_layer_output = self.attention.forward(first_layer_output)
        x = x + first_layer_output

        # compute output of second layer and add to x
        second_layer_output = self.rms_norm_2.forward(x)
        second_layer_output = self.ffn.forward(second_layer_output)
        x = x + second_layer_output

        return x
    

### Define full transformer language model

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
                 d_model : int, num_heads : int, d_ff : int, theta: float, device=None, dtype=None):
        super().__init__()

        # create embedding layer
        self.embedding_layer = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # create stack of transformer blocks
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype)
            for layer in range(num_layers)
        ])

        # create language model head
        self.lm_head = LMHead(vocab_size, d_model, device=device, dtype=dtype)
    
    def set_weights(self, embedding_weights : torch.Tensor,
                    layer_weights : list[dict[str, torch.Tensor]],
                    final_norm_weights : torch.Tensor, unembedding_weights : torch.Tensor
                    ):
        self.embedding_layer.set_weights(embedding_weights)

        for block, weights in zip(self.transformer_blocks, layer_weights):
            block.set_weights(**weights) # type: ignore

        self.lm_head.set_weights(final_norm_weights, unembedding_weights)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # get embeddings
        x = self.embedding_layer.forward(x)

        # pass through all transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x)

        # pass through language model head
        x = self.lm_head.forward(x)

        return x