import os, ctypes
from typing import Optional
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
    self.texts = []

  def load_corpus(self, path: str):
    if not os.path.exists(path): raise IOError(f"Corpus file does not exist: {path}")
    self.texts = []
    with open(path, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if line: self.texts.append(line)
    print(f"Loaded {len(self.texts)} lines from corpus")

  def train(self, num_iterations: int = 10) -> int:
    if not self.texts: raise RuntimeError("No texts loaded. Call load_corpus() first.")
    text_array = (ctypes.c_char_p * len(self.texts))(*[t.encode('utf-8') for t in self.texts])
    result = lib.trainUnigram(self.trainer, text_array, len(self.texts), num_iterations)
    if not result: raise RuntimeError("Training failed")
    print(f"Training completed: {num_iterations} iterations performed.")
    return num_iterations

  def save(self, vocab_path: str):
    vocab_dir = os.path.dirname(vocab_path)
    if vocab_dir: os.makedirs(vocab_dir, exist_ok=True)
    if not lib.saveVocab(self.trainer, vocab_path.encode('utf-8')): raise RuntimeError(f"Failed to save vocabulary to {vocab_path}")
    print(f"Vocabulary saved to: {vocab_path}")

  def destroy(self):
    if getattr(self, "trainer", None):
      try: lib.trainerDestroy(self.trainer)
      finally: self.trainer = None

  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb): self.destroy()
  def __del__(self):
    try: self.destroy()
    except Exception: pass