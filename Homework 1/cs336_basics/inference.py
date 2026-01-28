import torch
from .bpe_tokenizer import BPETokenizer
from .transformer import Transformer, softmax
from einops import rearrange

def generate_text(model : Transformer, tokenizer : BPETokenizer,
                  prompt : str, max_tokens_generated : int, temperature : float = 1.0, top_p : float = 0.95) -> str:
    # create initial prompt
    tokens : torch.Tensor = torch.Tensor(tokenizer.encode(prompt))
    tokens = torch.cat([torch.Tensor([0]), tokens], dim=-1) # start with EOS token
    tokens = tokens.to(torch.long)

    # generate new tokens
    model.eval()
    for i in range(max_tokens_generated):
        # get logits
        logits = model.forward(tokens)[-1]

        # get probabilities
        probabilities = softmax(logits, temperature=temperature)

        # perform top-p filtering
        sorted_probs, prob_indices = torch.sort(probabilities, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        exclusive_probs = torch.zeros_like(cumulative_probs)
        exclusive_probs[..., 1:] = cumulative_probs[..., :-1]
        top_probs = sorted_probs * (exclusive_probs <= top_p)

        # pick random token
        next_token_idx = torch.multinomial(top_probs, num_samples=1)
        next_token = prob_indices[next_token_idx]

        # if end, stop inference
        if next_token.item() == 0:
            break
        else:
            tokens = torch.cat([tokens, next_token], dim=-1)

    return tokenizer.decode(tokens[1:].tolist())