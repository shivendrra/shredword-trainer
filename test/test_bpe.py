import os
import tempfile
import pytest
from shredword.trainer import BPETrainer

@pytest.fixture
def small_corpus(tmp_path):
  p = tmp_path / "corpus.txt"
  sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing involves computational linguistics.",
    "Deep learning models require large amounts of training data.",
    "Tokenization is an important preprocessing step in NLP.",
    "Subword tokenization helps handle out-of-vocabulary words.",
    "Byte pair encoding and SentencePiece are popular tokenization methods.",
    "Transformer models have revolutionized natural language understanding.",
    "BERT, GPT, and T5 are examples of pre-trained language models.",
    "Fine-tuning allows adapting pre-trained models to specific tasks.",
    "The attention mechanism enables models to focus on relevant parts.",
    "Positional encoding helps models understand sequence order.",
    "Multi-head attention processes different representation subspaces.",
    "Layer normalization stabilizes training in deep networks.",
    "Dropout prevents overfitting by randomly zeroing activations.",
    "Gradient descent optimizes model parameters during training.",
    "Backpropagation computes gradients for parameter updates.",
    "Cross-entropy loss is commonly used for classification tasks.",
    "Regularization techniques prevent models from memorizing training data.",
    "Evaluation metrics measure model performance on test datasets."
  ] * 100

  p.write_text("\n".join(sample_texts) + "\n", encoding="utf-8")
  return str(p)

def test_bpe_train_and_save(small_corpus, tmp_path):
  model = tmp_path / "bpe.model"
  vocab = tmp_path / "bpe.vocab"

  trainer = BPETrainer(
    vocab_size=300,
    min_pair_freq=2
  )
  trainer.load_corpus(small_corpus)
  merges = trainer.train()

  assert merges > 0

  trainer.save(str(model), str(vocab))
  trainer.destroy()

  assert model.exists()
  assert vocab.exists()
  assert model.stat().st_size > 0
  assert vocab.stat().st_size > 0

def test_bpe_zero_merge_expected(small_corpus):
  trainer = BPETrainer(
    vocab_size=50,
    min_pair_freq=1000
  )
  trainer.load_corpus(small_corpus)
  merges = trainer.train()
  trainer.destroy()

  assert merges == 0

def test_bpe_rejects_missing_corpus(tmp_path):
  trainer = BPETrainer(vocab_size=10)
  with pytest.raises(IOError):
    trainer.load_corpus(str(tmp_path / "missing.txt"))

if __name__ == "__main__":
  pytest.main([__file__, "-v"])