"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, texts, vocab_size, verbose=False):
        # input text preprocessing
        all_bytes = []
        for text in texts:
            all_bytes.extend(text.encode("utf-8"))  # raw bytes
        unique_bytes = set(all_bytes)
        
        # Create initial vocabulary from unique bytes in the training data
        self.byte_to_id = {byte: i for i, byte in enumerate(unique_bytes)}
        self.id_to_byte = {i: byte for byte, i in self.byte_to_id.items()}
        
        ids = [self.byte_to_id[byte] for byte in all_bytes]
        
        initial_vocab_size = len(unique_bytes)
        num_merges = vocab_size - initial_vocab_size
        print(f"Training BPE with {initial_vocab_size} unique bytes and {num_merges} merges")

        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([self.id_to_byte[idx]]) for idx in range(initial_vocab_size)}
        
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            if not stats:
                break  # No more pairs to merge
            
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = initial_vocab_size + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    def decode(self, ids_list):
        # given a list of lists of integers, return a list of Python strings
        texts = []
        for ids in ids_list:
            text_bytes = b"".join(self.vocab[idx] for idx in ids)
            text = text_bytes.decode("utf-8", errors="replace")
            texts.append(text)
        return texts

    def encode(self, texts):
        # given a list of strings, return a list of token ids for each string
        all_ids = []
        for text in texts:
            text_bytes = text.encode("utf-8")  # raw bytes
            ids = [self.byte_to_id.get(byte, len(self.byte_to_id)) for byte in text_bytes]  # Use a default value for unknown bytes
            
            while len(ids) >= 2:
                # find the pair with the lowest merge index
                stats = get_stats(ids)
                if not stats:
                    break  # No more pairs to merge
                
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                # subtle: if there are no more merges available, the key will
                # result in an inf for every single pair, and the min will be
                # just the first pair in the list, arbitrarily
                # we can detect this terminating case by a membership check
                if pair not in self.merges:
                    break  # nothing else can be merged anymore
                # otherwise let's merge the best pair (lowest merge index)
                idx = self.merges[pair]
                ids = merge(ids, pair, idx)
            all_ids.append(ids)
        return all_ids