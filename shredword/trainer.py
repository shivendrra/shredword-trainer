import os, ctypes
from typing import Optional
from .cbase import lib, BPEConfig

class BPETrainer:
  def __init__(self, vocab_size=8192, unk_id=0, character_coverage=0.995, min_pair_freq=2000):
    self.config = BPEConfig(target_vocab_size=vocab_size, unk_id=unk_id, character_coverage=character_coverage, min_pair_freq=min_pair_freq)
    self.trainer = lib.create_trainer(ctypes.byref(self.config))
    if not self.trainer: raise RuntimeError("Failed to create BPE trainer")
    self._load_corpus = lib.bpe_load_corpus
    self._train = lib.bpe_train
    self._save = lib.bpe_save
    self._destroy_fn = lib.bpe_trainer_destroy

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
    model_dir = os.path.dirname(model_path)
    vocab_dir = os.path.dirname(vocab_path)
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