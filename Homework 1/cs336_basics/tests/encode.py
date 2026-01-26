from ..bpe_tokenizer import BPETokenizer
import time

# serialization_path = "data/pickled_data/TinyStoriesV2-GPT4-train.txt.pkl"
# encoding_path = "data/TinyStoriesV2-GPT4-valid.txt"
serialization_path = "data/pickled_data/owt_train.txt.pkl"
encoding_path = "data/owt_valid.txt"

# set up tokenizer
bpe_tokenizer = BPETokenizer()
special_tokens = ["<|endoftext|>"]
bpe_tokenizer.from_file(serialization_path, special_tokens)

# use first validation documents for encoding
num_documents = 1
full_file : str
with open(encoding_path, "r") as file:
    full_file = file.read()

position = -1
for i in range(num_documents):
    position = full_file.find("<|endoftext|>", position+1) + len("<|endoftext|>") - 1
    if position == -1:
        raise Exception("Error: Not enough documents.")

full_file = full_file[:(position+1)]

# find number of tokens and bytes for all documents
full_file_bytes = full_file.encode("utf-8")
start = time.time()
full_file_tokens = bpe_tokenizer.encode(full_file)
end = time.time()

print(f"Bytes: {len(full_file_bytes)}")
print(f"Tokens: {len(full_file_tokens)}")
print(f"Time: {end - start}")