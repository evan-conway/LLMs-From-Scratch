from ..bpe_tokenizer import BPETokenizer

bpe_tokenizer = BPETokenizer()

input_path = "data/TinyStoriesV2-GPT4-valid.txt"
# input_path = "data/owt_train.txt"
# input_path = "tests/fixtures/corpus.en"
vocab_size = 500
special_tokens = ["<|endoftext|>"]

bpe_tokenizer.train(input_path, vocab_size, special_tokens) # type: ignore

# print token merging information
print("\n\n")
print("MERGES")
for merge_count, (first_token_bytes, second_token_bytes) in enumerate(bpe_tokenizer.merges):
    first_token_text = first_token_bytes.decode("utf-8", errors="ignore")
    second_token_text = second_token_bytes.decode("utf-8", errors="ignore")
    new_token_text = (first_token_bytes + second_token_bytes).decode("utf-8", errors="ignore")
    print(f'{merge_count}: "{first_token_text}" + "{second_token_text}" = "{new_token_text}"')

# print vocabulary information
print("\n\n")
print("VOCABULARY")
for token_number, token_bytes in bpe_tokenizer.vocab.items():
    token_text = token_bytes.decode("utf-8", errors="ignore")
    print(f'{token_number}: "{token_text}"')

# print longest token
print("\n\n")
print("VOCABULARY STATS")
longest_token = max(bpe_tokenizer.vocab.values(), key=lambda x: len(x)).decode("utf-8", errors="ignore")
print(f'Longest token: "{longest_token}"')