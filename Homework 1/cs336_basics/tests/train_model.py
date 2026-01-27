from .. import training
from ..transformer import Transformer
import argparse
import numpy as np
import wandb
import os
import torch

# specify training device
DEVICE = "cpu"

# specify all model-related constants
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 512
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000

# specify training loop constants
BATCH_SIZE = 10
TRAINING_ITERATIONS = 1000 // BATCH_SIZE
SERIALIZATION_FREQUENCY = 10

# specify all optimizer constants
MAX_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-6

WARMUP_ITERATIONS = int(0.03 * TRAINING_ITERATIONS)
ANNEALING_ITERATIONS = TRAINING_ITERATIONS

BETA_1 = 0.9
BETA_2 = 0.98
WEIGHT_DECAY = 0.01

# initialize transformer model
transformer = Transformer(vocab_size=VOCAB_SIZE,
                          context_length=CONTEXT_LENGTH,
                          num_layers=NUM_LAYERS,
                          d_model=D_MODEL,
                          num_heads=NUM_HEADS,
                          d_ff=D_FF,
                          theta=ROPE_THETA,
                          device=DEVICE)

# initialize optimizer
optimizer = training.AdamW(params=transformer.parameters(),
                           lr=MAX_LEARNING_RATE,
                           betas=(BETA_1, BETA_2),
                           weight_decay=WEIGHT_DECAY)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--save", type=str, default=None)
parser.add_argument("--disable_wandb", action="store_true")
parser.add_argument("--wandb", type=str, default=None)
args = parser.parse_args()

# load data
file_size = os.path.getsize(args.data)
num_elements = file_size // np.dtype("uint16").itemsize
data = np.memmap(args.data, dtype="uint16", mode="r", shape=(num_elements,))

# if load path is given, load into transformer and optimizer
if args.load is not None:
    start_iteration = training.load_checkpoint(args.load, transformer, optimizer)
    print(f"loaded model")
else:
    start_iteration = 0

# initialize wandb
if not args.disable_wandb:
    run = wandb.init( # type: ignore
        entity="turbocon501-university-of-virginia",
        project="llms-from-scratch",
        id=args.wandb,
        resume="allow"
    )

# run training loop
torch.autograd.set_detect_anomaly(True)
for iteration in range(start_iteration, TRAINING_ITERATIONS):
    # update learning rate
    learning_rate = training.cosine_learning_rate(iteration + 1,
                                                  lr_max=MAX_LEARNING_RATE,
                                                  lr_min=MIN_LEARNING_RATE,
                                                  warmup_time=WARMUP_ITERATIONS,
                                                  annealing_time=ANNEALING_ITERATIONS)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # fetch batch
    inputs, outputs = training.load_data(data=data,
                                         batch_size=BATCH_SIZE,
                                         context_length=CONTEXT_LENGTH,
                                         device=DEVICE)
    inputs = inputs.to(torch.long)
    outputs = outputs.to(torch.long)

    # initialize model for training
    transformer.train()
    optimizer.zero_grad(set_to_none=True)

    # check if things need to be serialized, and serialize them if so
    if iteration % SERIALIZATION_FREQUENCY == 0 and args.save is not None:
        training.save_checkpoint(transformer, optimizer, iteration, args.save)
        print(f"{iteration}: saved model")

    # forward pass
    logits = transformer.forward(inputs)
    loss = training.cross_entropy_loss(logits, outputs)

    # log metrics
    print(f"{iteration}: training loss {loss}")
    if not args.disable_wandb:
        run.log({ # type: ignore
            "training loss": loss,
            "learning rate": learning_rate
            })

    # backward pass
    loss.backward()
    optimizer.step()