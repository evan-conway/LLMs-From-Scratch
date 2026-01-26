from ..bpe_tokenizer import BPETokenizer
from ..pretokenization_example import find_chunk_boundaries
import time
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
from itertools import accumulate
from operator import add
from itertools import chain

serialization_path = "data/pickled_data/TinyStoriesV2-GPT4-train.txt.pkl"
encoding_path = "data/TinyStoriesV2-GPT4-train.txt"
output_path = "data/serialized_data/TinyStoriesV2-GPT4-train.dat"

# serialization_path = "data/pickled_data/owt_train.txt.pkl"
# encoding_path = "data/owt_train.txt"
# output_path = "data/serialized_data/owt_train.dat"

# set up tokenizer
bpe_tokenizer = BPETokenizer()
special_tokens = ["<|endoftext|>"]
bpe_tokenizer.from_file(serialization_path, special_tokens)

# compute all chunk boundaries
file = open(encoding_path, "rb")
num_processes = mp.cpu_count() * 8
boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
file.close()

print("Found Boundaries")

# get exact token count
token_counts = [0] * num_processes

def read_and_count(index, start, end):
    with open(encoding_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
    tokens = bpe_tokenizer.encode(chunk)
    return (index, len(tokens))

def count_tokens(return_val):
    index, token_count = return_val
    token_counts[index] = token_count
    print(f"finished chunk {index}, counted {token_count} tokens")

with mp.Pool(mp.cpu_count()) as pool:
    # perform pretokenization on each chunk, getting the counts for each pretoken
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        pool.apply_async(read_and_count, args=(i, start, end,), callback=count_tokens, error_callback=print)

    # get results
    pool.close()
    pool.join()

# compute cumulative sums and create array

cumulative_token_counts = list(chain([0], accumulate(token_counts[:-1], add)))
print(f"Counted {sum(token_counts)} total tokens")
overall_tokens : npt.NDArray[np.uint16] = np.memmap(output_path, dtype="uint16", mode="w+", shape=(sum(token_counts),))

# process tokens again, but write to array this time

def read_and_encode(index, start, end):
    with open(encoding_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
    tokens = bpe_tokenizer.encode(chunk)
    return (index, tokens)

def write_tokens(return_val):
    index, tokens = return_val
    start = cumulative_token_counts[index]
    end = start + token_counts[index]
    overall_tokens[start:end] = np.array(tokens, dtype="uint16")
    print(f"finished chunk {index}")

with mp.Pool(mp.cpu_count()) as pool:
    # perform pretokenization on each chunk, getting the counts for each pretoken
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        pool.apply_async(read_and_encode, args=(i, start, end,), callback=write_tokens, error_callback=print)

    # get results
    pool.close()
    pool.join()

print("Serialized all tokens to numpy array")