import os
import struct
import pytest
from shredword import UnigramTrainer

MAGIC = 0x554E4752

@pytest.fixture
def small_corpus(tmp_path):
  p = tmp_path / "corpus.txt"
  p.write_text(
    "low lower lowest\n"
    "newer wider\n"
    "tokenization test\n"
  )
  return str(p)

def test_unigram_train_and_binary_save(small_corpus, tmp_path):
  model = tmp_path / "unigram.model"

  trainer = UnigramTrainer(vocab_size=50)
  trainer.load_corpus(small_corpus)
  trainer.train(num_iterations=3)
  trainer.save(str(model))
  trainer.destroy()

  assert model.exists()
  assert model.stat().st_size > 16

  with open(model, "rb") as f:
    magic, version, count = struct.unpack("<III", f.read(12))

  assert magic == MAGIC
  assert version == 1
  assert count > 0

def test_unigram_no_corpus_error():
  trainer = UnigramTrainer(vocab_size=10)
  with pytest.raises(RuntimeError):
    trainer.train()

def test_unigram_deterministic_small_run(small_corpus, tmp_path):
  m1 = tmp_path / "a.model"
  m2 = tmp_path / "b.model"

  t1 = UnigramTrainer(vocab_size=30)
  t1.load_corpus(small_corpus)
  t1.train(2)
  t1.save(str(m1))
  t1.destroy()

  t2 = UnigramTrainer(vocab_size=30)
  t2.load_corpus(small_corpus)
  t2.train(2)
  t2.save(str(m2))
  t2.destroy()

  assert m1.read_bytes() == m2.read_bytes()

if __name__ == "__main__":
  pytest.main([__file__, "-v"])