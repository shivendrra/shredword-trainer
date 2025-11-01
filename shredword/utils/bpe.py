import regex as re
import unicodedata
from collections import deque, Counter

merges = {}
vocab = {idx: bytes([idx]) for idx in range(256)}
pattern = ""
special_tokens = {}

def get_stats(ids, counts=None):
  """
    takes list of integers and returns dictionary of counts of pairs(consecutive ones)
    eg: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    allows to update an existing dictionary of counts
  """
  # counts = {} if counts is None else counts
  # for pair in zip(ids, ids[1:]):
  #   counts[pair] = counts.get(pair, 0) + 1
  # return counts
  return Counter(zip(ids, ids[1:])) # using Counter over the previous code logic is better as it's optimzed for task like this

def merge(ids, pair, idx):
  """
    in the list of integers, replaces all consecutive pair with the new integer token idx
    eg: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
  """
  # merged = []
  merged, i = deque(), 0 # replaced [] -> deque to minimize the memory reallocations
  while i < len(ids):
    if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
      merged.append(idx)
      i += 2
    else:
      merged.append(ids[i])
      i += 1
  return list(merged)

def apply_regex(text):
  r"""
  	## space is merged with each word, before it as a prefix
  	## a litlle smaller than pattern2
	  regex_pattern1: '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
	
	  ## space is added as a preffix to each word, retains all the initial words
	  ## smaller than pattern3
  	regex_pattern2: '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

  	## space is considered a separate token, all words remain original, no loss of words
  	## largest in length
  	regex_pattern3: 's|'t|'re|'ve|'m|'ll|'d|[\w']+|[^\s\w\d]+|\s+(?!\S)|\s+
  
  	## spaces are added as a prefix to the words, but some words are missing hence doesn't retains original text
  	## smallest in length, due to some lost words
  	regex_pattern4: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+ | ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
	"""
  pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
  text = re.findall(pattern, text)
  return text

def build_vocab(merges, special_tokens):
  """
    ## this function basically builds the primary vocab (0-255)
    ## uses 256-ascii characters & put them into a key-value paired dictonary to form
    a lookup table to build merges & get stats of total pairs of bytes
    so the base vocab looks something like this: {'!':0, 'a': 1, 'b': 2, 'c':3 ....., 'x03':255}

    ## uses provided merges & adds new entries to original vocab & also incorporates the 
    special tokens: <|endoftext|>, <|mask|>, <|startoftext|>, etc.
    basically, builds a map of each byte pair to a corresponding integer value/representation

      merges = {('a', 'b'): 256, ('c', 'd'): 257}
      special_tokens = {'<pad>': 258, '<unk>': 259}
  """
  vocab = {idx: bytes([idx]) for idx in range(256)}
  for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
  for special, idx in special_tokens.items():
    vocab[idx] = special.encode("utf-8")
  return vocab

def replace_control_characters(s: str) -> str:
  # we don't want to print control characters
  # which distort the output (e.g. \n)
  chars = []
  for ch in s:
    if unicodedata.category(ch)[0] != "C":
      chars.append(ch) # this character is ok
    else:
      chars.append(f"\\u{ord(ch):04x}") # escape
  return "".join(chars)

def render_token(t: bytes) -> str:
  # pretty print a token, escaping control characters
  s = t.decode('utf-8', errors='replace')
  s = replace_control_characters(s)
  return s

class BaseTokenizer:
  def __init__(self):
    # default: vocab size of 256 (all bytes), no merges, no patterns
    self.merges = {} # (int, int) -> int
    self.pattern = ""
    self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
    self.vocab = build_vocab(self.merges, self.special_tokens) # int -> bytes

  # placeholder functions, implemented in child class
  def train(self, text, vocab_size, verbose=False): raise NotImplementedError
  def encode(self, text): raise NotImplementedError
  def decode(self, ids): raise NotImplementedError

  def save(self, file_prefix):
    # saves two files: ``.model`` & ``.vocab``
    # `.vocab` human readable version thats just for debugging & pretty presentation
    # `.model` for furthur training & implementing merges, can be loaded in model
    model_file = file_prefix + ".model"
    with open(model_file, 'w') as f:
      f.write("shredword v1\n")
      f.write(f"{self.pattern}\n")
      f.write(f"{len(self.special_tokens)}\n")
      for special, idx in self.special_tokens.items():
        f.write(f"{special} {idx}\n")
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")
    vocab_file, inverted_merges = file_prefix + ".vocab", {idx: pair for pair, idx in self.merges.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
      for idx, token in self.vocab.items():
        s = render_token(token)
        if idx in inverted_merges:
          idx0, idx1 = inverted_merges[idx]
          s0, s1 = render_token(self.vocab[idx0]), render_token(self.vocab[idx1])
          f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
        else: f.write(f"[{s}] {idx}\n")
      f.close()

  def load(self, model_file):
    assert model_file.endswith(".model")
    merges, special_tokens, idx = {}, {}, 256
    with open(model_file, 'r', encoding="utf-8") as f:
      version = f.readline().strip()
      assert version == "shredword v1"
      self.pattern, num_special = f.readline().strip(), int(f.readline().strip())
      for _ in range(num_special):
        special, special_idx = f.readline().strip().split()
        special_tokens[special] = int(special_idx)
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    self.merges, self.special_tokens, self.vocab = merges, special_tokens, build_vocab(merges, special_tokens)

class BPETokenizer(BaseTokenizer):
  def __init__(self, pattern=None):
    super().__init__()
    # default GPT-4 pattern for pre-tokenization if none provided
    self.pattern = pattern or r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    self.compiled_pattern = re.compile(self.pattern)

  def train(self, text, vocab_size, verbose=False):
    """
      trains BPE tokenizer on given text to reach desired vocab_size
      starts with base vocab of 256 bytes, then iteratively merges most frequent pairs
      until vocab_size is reached
    """
    assert vocab_size >= 256
    num_merges = vocab_size - 256
    text_chunks = re.findall(self.compiled_pattern, text) # pre-tokenize the text using regex pattern
    ids = [list(ch.encode("utf-8")) for ch in text_chunks]      # convert each chunk to list of bytes (UTF-8 encoded)

    # iteratively merge the most frequent pairs
    merges = {} # (int, int) -> int
    vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    for i in range(num_merges):
      # get statistics of all pairs across all chunks
      stats = {}
      for chunk_ids in ids:
        chunk_stats = get_stats(chunk_ids)
        for pair, count in chunk_stats.items(): stats[pair] = stats.get(pair, 0) + count
      # find the pair with highest count
      if not stats: break # no more pairs to merge
      pair = max(stats, key=stats.get)
      idx = 256 + i # new token id
      if verbose: print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({stats[pair]} occurrences)")
      ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]        # merge the pair in all chunks
      # save the merge & update vocab
      merges[pair] = idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    # save learned merges & vocab
    self.merges = merges
    self.vocab = vocab

  def _encode_chunk(self, text_bytes):
    """encode a single chunk of text bytes using learned merges"""
    ids = list(text_bytes)
    while len(ids) >= 2:
      stats = get_stats(ids)  # get all pair statistics
      pair = min(stats, key=lambda x: self.merges.get(x, float("inf"))) # find the pair with lowest merge index (earliest learned)
      if pair not in self.merges: break # if pair not in merges, we can't merge anymore
      # merge the pair
      idx = self.merges[pair]
      ids = merge(ids, pair, idx)
    return ids

  def encode(self, text):
    """encode text into list of token ids using BPE"""
    # handle special tokens first if any
    # for now, simple implementation without special token handling
    # pre-tokenize using regex
    text_chunks = re.findall(self.compiled_pattern, text)
    # encode each chunk
    ids = []
    for chunk in text_chunks:
      chunk_bytes = chunk.encode("utf-8")
      chunk_ids = self._encode_chunk(chunk_bytes)
      ids.extend(chunk_ids)
    return ids

  def decode(self, ids):
    """decode list of token ids back to text"""
    # convert tokens to bytes
    text_bytes = b""
    for idx in ids:
      if idx in self.vocab: text_bytes += self.vocab[idx]
      else: raise ValueError(f"invalid token id: {idx}")
    # decode bytes to text
    text = text_bytes.decode("utf-8", errors="replace")
    return text