from .pretokenization_example import find_chunk_boundaries
import regex as re
import heapq
from collections import defaultdict
from collections.abc import Iterator, Iterable
import multiprocessing as mp
import pickle
import time
from functools import cache

class ReverseBytes:
    def __init__(self, word):
        self.word = word

    def __lt__(self, other):
        return self.word > other.word

    def __eq__(self, other):
        return self.word == other.word

class BPETokenizer:
    vocab : dict[int, bytes]
    inverse_vocab : dict[bytes, int]
    merges : list[tuple[bytes, bytes]]
    special_tokens : list[str]

    def __init__(self, vocab : dict[int, bytes] | None = None, merges : list[tuple[bytes, bytes]]  | None = None, special_tokens : list[str] | None = None):
        if special_tokens is None:
            special_tokens = []

        if vocab is not None and merges is not None:
            self.vocab = vocab
            self.inverse_vocab = dict()
            for token_num, token_bytes in self.vocab.items():
                self.inverse_vocab[token_bytes] = token_num

            self.merges = merges
            self.inverse_merges = dict()
            for merge_num, merge in enumerate(self.merges):
                self.inverse_merges[merge] = merge_num

            self.special_tokens = special_tokens
        else:
            self.initialize_vocab(special_tokens)

    def initialize_vocab(self, special_tokens : list[str]):
        # initialize vocabulary and inverse vocabulary
        self.vocab : dict[int, bytes] = dict()
        self.inverse_vocab : dict[bytes, int] = dict()

        # add special tokens
        for token in special_tokens:
            token_num = len(self.vocab)
            token_bytes = token.encode("utf-8")
            self.vocab[token_num] = token_bytes
            self.inverse_vocab[token_bytes] = token_num
        
        # add initial bytes
        for int_value in range(256):
            token_num = len(self.vocab)
            token_bytes = int_value.to_bytes(1, byteorder="big")
            self.vocab[token_num] = token_bytes
            self.inverse_vocab[token_bytes] = token_num

        # initialize merges
        self.merges : list[tuple[bytes, bytes]] = []
        self.inverse_merges : dict[tuple[bytes, bytes], int] = dict()

        # initialize special tokens
        self.special_tokens = special_tokens

    def from_file(self, serialization_path : str, special_tokens : list[str]):
        """ Load in serialized vocab and merge data.
        
        Args:
            serialization_path: Path to load serialized data from.
            special_tokens: List of all special tokens.
        """


        # initialize vocab, inverse vocab, and merges
        with open(serialization_path, "rb") as file:
            self.merges, self.vocab = pickle.load(file)
        
        self.inverse_vocab = dict()
        for token_num, token_bytes in self.vocab.items():
            self.inverse_vocab[token_bytes] = token_num

        self.inverse_merges = dict()
        for merge_num, merge in enumerate(self.merges):
            self.inverse_merges[merge] = merge_num

        # initialize special tokens
        self.special_tokens = special_tokens

    def pretokenize_chunk(self, chunk : str) -> Iterator[str]:
        """ Performs pretokenization for the given chunk.

        Args:
            chunk: Text to pretokenize.
        
        Yields:
            Pretokens within the chunk.
        """

        # split into different documents
        if len(self.special_tokens) > 0:
            escaped_special_tokens = [re.escape(token) for token in sorted(self.special_tokens, key=lambda x: len(x), reverse=True)]
            delimiter = r"(%s)" % "|".join(escaped_special_tokens)
            documents = re.split(delimiter, chunk)
        else:
            documents = [chunk]

        pretokenization_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for document in documents:
            if document in self.special_tokens:
                yield document
            else:
                for pretoken_match in re.finditer(pretokenization_regex, document):
                    yield pretoken_match.group()

    def count_pretokens_in_chunk(self, chunk : str) -> dict[tuple[int], int]:
        """ Count the number of each pretoken in a given chunk.
        
        Args:
            chunk: Text to pretokenize.

        Returns:
            Counts for each pretoken in the chunk, where the pretoken is a tuple of tokens.
        """

        # track counts for each pretoken
        pretoken_counts : defaultdict[str, int] = defaultdict(int)

        # iterate over pretokens in each document, adding to overall pretoken counts
        for pretoken_string in self.pretokenize_chunk(chunk):
            pretoken_counts[pretoken_string] += 1

        # convert each pretoken to lists of tokens
        pretoken_counts_tokenized : dict[tuple[int], int] = dict()
        for pretoken, count in pretoken_counts.items():
            pretoken_bytes = pretoken.encode("utf-8")

            # if this can be mapped to a specific token, do so
            if pretoken_bytes in self.inverse_vocab:
                pretoken_tokens = (self.inverse_vocab[pretoken_bytes],)
            # otherwise, convert all bytes to the correct tokens
            else:
                pretoken_tokens = tuple(self.inverse_vocab[bytes([byte])] for byte in pretoken_bytes)
            pretoken_counts_tokenized[pretoken_tokens] = count # type: ignore

        return pretoken_counts_tokenized

    def train(self, input_path : str, vocab_size : int, special_tokens : list[str], serialization_path : str | None = None):
        """ Train tokenizer on documents, training to a specific vocabulary size.
        
        Args:
            input_path: Path of the documents to train on.
            vocab_size: Size of the vocabulary, including special tokens.
            special_tokens: List of all special tokens.
            serialization_path: Path to save serialized data to.
        """

        self.initialize_vocab(special_tokens)

        combined_pretoken_counts : defaultdict[tuple[int], int] = defaultdict(int)

        # compute all chunk boundaries
        file = open(input_path, "rb")
        num_processes = 5 * mp.cpu_count()
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
        file.close()

        # get pretoken counts across all chunks
        def combine_pretokens(pretoken_counts):
            # combine pretoken counts
            for pretoken, count in pretoken_counts.items():
                combined_pretoken_counts[pretoken] += count

        with mp.Pool(mp.cpu_count()) as pool, open(input_path, "rb") as file:
            # perform pretokenization on each chunk, getting the counts for each pretoken
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                # read chunk information
                file.seek(start)
                chunk = file.read(end - start).decode("utf-8", errors="ignore")
                pool.apply_async(self.count_pretokens_in_chunk, args=(chunk,), callback=combine_pretokens, error_callback=print)

            # get results
            pool.close()
            pool.join()

        # convert pretokens into a list, which gives each a unique identifier
        pretoken_counts_list : list[tuple[tuple[int], int]] = []
        for pretoken, count in combined_pretoken_counts.items():
            pretoken_counts_list.append((pretoken, count))

        # get count for each pair and index pretokens with each pair
        token_pair_counts : defaultdict[tuple[int, int], int] = defaultdict(int)
        token_pair_references : defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for pretoken_idx, (pretoken, count) in enumerate(pretoken_counts_list):
            # iterate over all pairs of tokens
            for first_token, second_token in zip(pretoken, pretoken[1:]):
                pair = (first_token, second_token)

                # increase count of token, and add this to its appearances
                token_pair_counts[pair] += count
                token_pair_references[pair].add(pretoken_idx)

        # define heap for token pairs
        token_pair_heap = []

        def add_to_heap(token_pair, count):
            first_token_bytes_reversed = ReverseBytes(self.vocab[token_pair[0]])
            second_token_bytes_reversed = ReverseBytes(self.vocab[token_pair[1]])
            heapq.heappush(token_pair_heap, (-count, (first_token_bytes_reversed, second_token_bytes_reversed), token_pair)) # must invert for max heap

        def remove_from_heap():
            neg_count, reversed_combination, most_common_pair = heapq.heappop(token_pair_heap)
            count = -neg_count
            return count, most_common_pair

        # convert token pair counts into a max heap
        for token_pair, count in token_pair_counts.items():
            add_to_heap(token_pair, count)
        
        # keep merging until vocabulary as at the desired size
        token_pair_deltas : defaultdict[tuple[int, int], int] = defaultdict(int)
        while len(self.vocab) < vocab_size:
            # get next merge
            most_common_pair : tuple[int, int]
            while True:
                # get the most common pair of tokens
                count, most_common_pair = remove_from_heap()

                # if the count is correct, stop
                if token_pair_deltas[most_common_pair] == 0:
                    break
                # otherwise, update the count and push back
                else:
                    count -= token_pair_deltas[most_common_pair]
                    add_to_heap(most_common_pair, count)
                    token_pair_deltas[most_common_pair] = 0

            # add new token and merge to vocabulary
            new_token = len(self.vocab)
            first_token_bytes = self.vocab[most_common_pair[0]]
            second_token_bytes = self.vocab[most_common_pair[1]]
            new_token_bytes = first_token_bytes + second_token_bytes
            merge = (first_token_bytes, second_token_bytes)

            self.vocab[new_token] = new_token_bytes
            self.inverse_vocab[new_token_bytes] = new_token
            self.merges.append(merge)
            self.inverse_merges[merge] = len(self.merges) - 1
            
            # get the list of relevant pretokens to update
            pretokens_to_update = token_pair_references[most_common_pair]
            del token_pair_references[most_common_pair]
        
            # iterate over all pretokens to update
            new_token_pair_counts : defaultdict[tuple[int, int], int] = defaultdict(int)
            for pretoken_idx in pretokens_to_update:
                pretoken, count = pretoken_counts_list[pretoken_idx]
                
                # get the new pretoken, new token pair counts, and old token pair deltas
                new_pretoken = []
                token_idx = 0
                while token_idx < len(pretoken):
                    # if this is an instance of the most common pair, replace it with the new token
                    if token_idx+1 < len(pretoken) and (pretoken[token_idx], pretoken[token_idx+1]) == most_common_pair:
                        # update token pair counts
                        if token_idx-1 >= 0:
                            old_pair = (pretoken[token_idx-1], pretoken[token_idx])
                            new_pair = (pretoken[token_idx-1], new_token)

                            new_token_pair_counts[new_pair] += count
                            token_pair_references[new_pair].add(pretoken_idx)
                            token_pair_deltas[old_pair] += count
                        if token_idx+2 < len(pretoken):
                            old_pair = (pretoken[token_idx+1], pretoken[token_idx+2])
                            new_pair = (new_token, pretoken[token_idx+2])

                            new_token_pair_counts[new_pair] += count
                            token_pair_references[new_pair].add(pretoken_idx)
                            token_pair_deltas[old_pair] += count
                        
                        # add token
                        new_pretoken.append(new_token)
                        token_idx += 2
                    # otherwise, transfer token over normally
                    else:
                        new_pretoken.append(pretoken[token_idx])
                        token_idx += 1

                # update with corrected pretoken tuple
                new_pretoken_tuple : tuple[int] = tuple(new_pretoken)
                pretoken_counts_list[pretoken_idx] = (new_pretoken_tuple, count)

            # add the new token pairs to the heap
            for token_pair, count in new_token_pair_counts.items():
                add_to_heap(token_pair, count)

        # save merges and vocabulary
        if serialization_path is not None:
            with open(serialization_path, "wb") as file:
                pickle.dump((self.merges, self.vocab), file)
    
    @cache
    def encode_pretoken(self, pretoken : str) -> list[int]:
        """ Encode pretoken into tokens.

        Args:
            pretoken: The pretoken to encode.

        Returns:
            The pretoken, tokenized using the vocabulary into a list of tokens.
        """

        pretoken_encoding : bytes = pretoken.encode("utf-8")

        # check to see if this corresponds exactly to a token
        if pretoken_encoding in self.inverse_vocab:
            return [self.inverse_vocab[pretoken_encoding]]
        else:
            # get the initial list of bytes in the pretoken
            pretoken_bytes : list[bytes] = [bytes([byte]) for byte in pretoken_encoding]

            # keep merging while there are merges to perform
            while True:
                # get next merge
                merge : tuple[bytes, bytes] | None = None
                lowest_merge_num = 10**10
                for first_bytes, second_bytes in zip(pretoken_bytes, pretoken_bytes[1:]):
                    candidate_merge = (first_bytes, second_bytes)
                    if candidate_merge in self.inverse_merges:
                        merge_num = self.inverse_merges[candidate_merge]
                        if merge_num < lowest_merge_num:
                            merge = candidate_merge
                            lowest_merge_num = merge_num

                # stop as soon as there is no merge we can apply
                if merge is not None:
                    pretoken_bytes_new = []
                    token_idx = 0
                    while token_idx < len(pretoken_bytes):
                        # if this is an instance of the most common pair, replace it with the new token
                        if token_idx+1 < len(pretoken_bytes) and (pretoken_bytes[token_idx], pretoken_bytes[token_idx+1]) == merge:
                            pretoken_bytes_new.append(pretoken_bytes[token_idx] + pretoken_bytes[token_idx+1])
                            token_idx += 2
                        # otherwise, transfer token over normally
                        else:
                            pretoken_bytes_new.append(pretoken_bytes[token_idx])
                            token_idx += 1

                    pretoken_bytes = pretoken_bytes_new
                else:
                    break

            # convert merged bytes to tokens
            pretoken_tokens : list[int] = [self.inverse_vocab[merged_bytes] for merged_bytes in pretoken_bytes]
        
            return pretoken_tokens

    def encode(self, text : str) -> list[int]:
        """ Encode text into tokens.

        Args:
            text: The text to encode.

        Returns:
            The text, tokenized using the vocabulary into a list of tokens.
        """

        tokenized_text : list[int] = []

        # perform pretokenization
        pretokenized_text = self.pretokenize_chunk(text)

        # encode pretokens into tokens
        for pretoken in pretokenized_text:
            pretoken_encoding = self.encode_pretoken(pretoken)
            tokenized_text.extend(pretoken_encoding)

        return tokenized_text
    
    def encode_iterable(self, texts : Iterable[str]) -> Iterator[int]:
        """ Lazily encode texts into tokens.

        Args:
            texts: The texts to encode.

        Yields:
            The texts, tokenized using the vocabulary into a list of tokens.
        """

        for text in texts:
            yield from self.encode(text)

    def decode(self, tokens : list[int]) -> str:
        """ Decodes a list of tokens into the corresponding string.

        Args:
            tokens: The list of tokens to decode.

        Returns:
            The string corresponding to the input tokens.
        """

        decoded_bytes : bytes = bytes()
        for token in tokens:
            decoded_bytes += self.vocab[token]

        decoded_string : str = decoded_bytes.decode("utf-8", errors="ignore")

        return decoded_string