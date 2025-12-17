import os, ctypes
from typing import Optional, List, Tuple, Dict
from .cbase import lib, BPEConfig

class BPETrainer:
  def __init__(self, vocab_size=8192, unk_id=0, character_coverage=0.995, min_pair_freq=2000):
    self.config = BPEConfig(target_vocab_size=vocab_size, unk_id=unk_id, character_coverage=character_coverage, min_pair_freq=min_pair_freq)
    self.trainer = lib.create_trainer(ctypes.byref(self.config))
    if not self.trainer: raise RuntimeError("Failed to create BPE trainer")
    self._load_corpus, self._train, self._save, self._destroy_fn = lib.bpe_load_corpus, lib.bpe_train, lib.bpe_save, lib.bpe_trainer_destroy

  def load_corpus(self, path: str):
    if not os.path.exists(path): raise IOError(f"Corpus file does not exist: {path}")
    result = self._load_corpus(self.trainer, path.encode('utf-8'))
    if result != 0: raise IOError(f"Failed to load corpus from {path} (code {int(result)})")

  def train(self) -> int:
    merges = self._train(self.trainer)
    if merges < 0: raise RuntimeError("Training failed")
    print(f"Training completed: {int(merges)} merges performed.")
    return int(merges)

  def save(self, model_path: str, vocab_path: str):
    model_dir, vocab_dir = os.path.dirname(model_path), os.path.dirname(vocab_path)
    if model_dir: os.makedirs(model_dir, exist_ok=True)
    if vocab_dir: os.makedirs(vocab_dir, exist_ok=True)
    self._save(self.trainer, model_path.encode('utf-8'), vocab_path.encode('utf-8'))
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")

  def destroy(self):
    if getattr(self, "trainer", None):
      try: self._destroy_fn(self.trainer)
      finally: self.trainer = None

  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb): self.destroy()
  def __del__(self):
    try: self.destroy()
    except Exception: pass


class UnigramTrainer:
  def __init__(self, vocab_size=32000, character_coverage=0.9995, max_sentencepiece_length=16, seed_size=1000000):
    self.vocab_size, self.character_coverage, self.max_len, self.seed_size = vocab_size, character_coverage, max_sentencepiece_length, seed_size
    self.trainer = lib.trainerCreate(vocab_size, character_coverage, max_sentencepiece_length, seed_size)
    if not self.trainer: raise RuntimeError("Failed to create Unigram trainer")
    self._destroy_fn = lib.trainerDestroy

  def add_text(self, text: str) -> bool:
    if not text: return False
    result = lib.addTextToTrainer(self.trainer, text.encode('utf-8'))
    if not result: raise RuntimeError("Failed to add text to trainer")
    return result

  def add_texts(self, texts: List[str]) -> int:
    count = 0
    for text in texts:
      if text and self.add_text(text): count += 1
    return count

  def preprocess(self) -> bool:
    result = lib.preprocessTexts(self.trainer)
    if not result: raise RuntimeError("Failed to preprocess texts")
    return result

  def extract_subwords(self) -> bool:
    result = lib.extractInitialSubwords(self.trainer)
    if not result: raise RuntimeError("Failed to extract initial subwords")
    return result

  def train(self, texts: Optional[List[str]] = None, num_iterations: int = 10) -> bool:
    if texts:
      text_array = (ctypes.c_char_p * len(texts))(*[t.encode('utf-8') for t in texts])
      result = lib.trainUnigram(self.trainer, text_array, len(texts), num_iterations)
    else:
      result = lib.trainUnigram(self.trainer, None, 0, num_iterations)
    if not result: raise RuntimeError("Training failed")
    print(f"Training completed: {num_iterations} iterations performed.")
    return result

  def compute_loss(self, texts: List[str]) -> float:
    if not texts: return 0.0
    text_array = (ctypes.c_char_p * len(texts))(*[t.encode('utf-8') for t in texts])
    return lib.computeLoss(self.trainer, text_array, len(texts))

  def compute_token_loss(self, token: str, texts: List[str]) -> float:
    if not texts or not token: return 0.0
    text_array = (ctypes.c_char_p * len(texts))(*[t.encode('utf-8') for t in texts])
    return lib.computeTokenLoss(self.trainer, token.encode('utf-8'), text_array, len(texts))

  def prune_vocab(self, texts: List[str], reduction_ratio: float = 0.8) -> bool:
    if not texts: return False
    text_array = (ctypes.c_char_p * len(texts))(*[t.encode('utf-8') for t in texts])
    result = lib.pruneVocabStep(self.trainer, text_array, len(texts), reduction_ratio)
    if not result: raise RuntimeError("Failed to prune vocabulary")
    return result

  def update_scores(self, texts: List[str]) -> bool:
    if not texts: return False
    text_array = (ctypes.c_char_p * len(texts))(*[t.encode('utf-8') for t in texts])
    result = lib.updateTokenScores(self.trainer, text_array, len(texts))
    if not result: raise RuntimeError("Failed to update token scores")
    return result

  def get_vocab(self) -> Dict[str, float]:
    tokens_ptr, scores_ptr, count = ctypes.POINTER(ctypes.c_char_p)(), ctypes.POINTER(ctypes.c_double)(), ctypes.c_int()
    result = lib.getVocab(self.trainer, ctypes.byref(tokens_ptr), ctypes.byref(scores_ptr), ctypes.byref(count))
    if not result: raise RuntimeError("Failed to get vocabulary")
    vocab = {}
    for i in range(count.value):
      token, score = tokens_ptr[i].decode('utf-8'), scores_ptr[i]
      vocab[token] = score
    return vocab

  def save(self, filepath: str):
    file_dir = os.path.dirname(filepath)
    if file_dir: os.makedirs(file_dir, exist_ok=True)
    result = lib.saveVocab(self.trainer, filepath.encode('utf-8'))
    if not result: raise RuntimeError(f"Failed to save vocabulary to {filepath}")
    print(f"Vocabulary saved to: {filepath}")

  def load(self, filepath: str):
    if not os.path.exists(filepath): raise IOError(f"Vocabulary file does not exist: {filepath}")
    result = lib.loadVocab(self.trainer, filepath.encode('utf-8'))
    if not result: raise IOError(f"Failed to load vocabulary from {filepath}")

  def destroy(self):
    if getattr(self, "trainer", None):
      try: self._destroy_fn(self.trainer)
      finally: self.trainer = None

  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb): self.destroy()
  def __del__(self):
    try: self.destroy()
    except Exception: pass