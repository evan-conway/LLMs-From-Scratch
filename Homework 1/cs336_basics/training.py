import torch
from einops import einsum, rearrange, repeat
from collections.abc import Callable
import math
import os
from typing import IO, Any, BinaryIO, Iterable
import numpy as np
import numpy.typing as npt
import random

def cross_entropy_loss(logits : torch.Tensor, targets : torch.Tensor):
    # get the max for the logits
    maxes = torch.max(logits, -1).values
    maxes = rearrange(maxes, "... -> ... 1")

    # shift by the max
    shifted = logits - maxes

    # raise all items to the exponential
    exponentials = torch.exp(shifted)

    # get sums
    sums = torch.sum(exponentials, -1)
    sums = rearrange(sums, "... -> ... 1")

    # get the target logits
    targets = rearrange(targets, "... -> ... 1")
    target_logits = torch.gather(shifted, -1, targets)

    # get cross entropy loss components
    return torch.mean(-(target_logits - torch.log(sums)))

def cosine_learning_rate(time : int, lr_min : float, lr_max : float, warmup_time : int, annealing_time : int) -> float:
    if time < warmup_time:
        return lr_max * (time / warmup_time)
    elif time <= annealing_time:
        return lr_min + (lr_max - lr_min) * (0.5 * (1 + math.cos(math.pi * (time - warmup_time)/(annealing_time - warmup_time))))
    else:
        return lr_min

def clip_gradients(params: Iterable[torch.nn.Parameter], max_norm: float, eps : float = 1e-6):
    norm : float = 0
    for param in params:
        if param.grad is not None:
            norm += torch.linalg.norm(param.grad)**2
    norm = math.sqrt(norm)

    if norm > max_norm:
        for param in params:
            if param.grad is not None:
                param.grad *= max_norm/(norm + eps)

def load_data(data : npt.NDArray[np.uint16], batch_size : int, context_length : int, device="cpu"):
    input_stack = []
    output_stack = []
    
    # sample batches
    for batch in range(batch_size):
        # choose starting address
        start = random.randint(0, len(data) - context_length - 1)
    
        # fetch relevant inputs and outputs
        inputs = torch.tensor(data[start:start+context_length], device=device)
        outputs = torch.tensor(data[start+1:start+context_length+1], device=device)

        # add inputs and outputs
        input_stack.append(inputs)
        output_stack.append(outputs)

    # stack tensors
    stacked_inputs = torch.stack(input_stack, dim=0)
    stacked_outputs = torch.stack(output_stack, dim=0)

    return (stacked_inputs, stacked_outputs)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure : Callable | None = None): # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        
        return 
    

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "epsilon": eps,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "decay": weight_decay
            }
        super().__init__(params, defaults)

    def step(self, closure : Callable | None = None): # type: ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # get group parameters
            lr = group["lr"]
            epsilon = group["epsilon"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            decay = group["decay"]
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # get state and state variables
                state = self.state[p]

                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                # get gradient of loss
                grad = p.grad.data
                
                # update moments
                state["m"] = m = beta_1 * m + (1 - beta_1) * grad
                state["v"] = v = beta_2 * v + (1 - beta_2) * grad**2

                # compute adjusted learning rate
                adjusted_lr = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)

                # update parameter with gradient descent and weight decay
                p.data -= adjusted_lr * m / (torch.sqrt(v) + epsilon)
                p.data *= 1 - lr * decay

                # increment iteration
                state["t"] = t + 1
        
        return loss
    

def save_checkpoint(model : torch.nn.Module, optimizer : torch.optim.Optimizer, iteration : int, out : str | os.PathLike | BinaryIO | IO[bytes]):
    # get parameters
    model_params = model.state_dict()
    optimizer_params = optimizer.state_dict()

    # save all information
    torch.save((model_params, optimizer_params, iteration), out)

def load_checkpoint(src : str | os.PathLike | BinaryIO | IO[bytes], model : torch.nn.Module, optimizer : torch.optim.Optimizer) -> int:
    # load all information
    model_params, optimizer_params, iteration = torch.load(src)

    # load parameters into model and optimizer
    model.load_state_dict(model_params)
    optimizer.load_state_dict(optimizer_params)

    return iteration