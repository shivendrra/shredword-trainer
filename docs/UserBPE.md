# BPE (Byte Pair Encoding) Tokenizer

## Overview

BPE is a subword tokenization algorithm that iteratively merges the most frequent pairs of bytes or characters. It starts with a character-level vocabulary and progressively builds larger subword units based on their frequency in the training corpus.

## Installation

Ensure you have the required dependencies installed and the underlying C library (`cbase`) is properly configured.

From [PyPI.org](https://pypi.org/project/shredword-trainer/):

```bash
pip install shredword-trainer
```

Importing:

```python
from shredword.trainer import BPETrainer
```

## Python API

### Quick Start

```python
trainer = BPETrainer()
trainer.load_corpus("path/to/your/corpus.txt")
trainer.train()
trainer.save("base.model", "base.vocab")
trainer.destroy()
```

### BPETrainer Class

#### Constructor

```python
BPETrainer(vocab_size=8192, unk_id=0, character_coverage=0.995, min_pair_freq=2000)
```

**Parameters:**
- `vocab_size` (int): Target vocabulary size. Default: 8192
- `unk_id` (int): ID for unknown tokens. Default: 0
- `character_coverage` (float): Character coverage ratio (0.0-1.0). Default: 0.995
- `min_pair_freq` (int): Minimum frequency for pair merging. Default: 2000

**Raises:**
- `RuntimeError`: If the trainer fails to initialize

#### Methods

##### load_corpus(path: str)

Loads training corpus from a text file.

**Parameters:**
- `path` (str): Path to corpus file

**Raises:**
- `IOError`: If file doesn't exist or fails to load

**Example:**
```python
trainer = BPETrainer(vocab_size=32000)
trainer.load_corpus("corpus.txt")
```

##### train() -> int

Trains the BPE model on loaded corpus.

**Returns:**
- `int`: Number of merges performed

**Raises:**
- `RuntimeError`: If training fails

**Example:**
```python
merges = trainer.train()
print(f"Completed {merges} merges")
```

##### save(model_path: str, vocab_path: str)

Saves trained model and vocabulary to files.

**Parameters:**
- `model_path` (str): Output path for model binary
- `vocab_path` (str): Output path for vocabulary file

**Example:**
```python
trainer.save("model.bin", "vocab.txt")
```

##### destroy()

Releases trainer resources. Called automatically on context exit or deletion.

#### Context Manager Support

```python
with BPETrainer(vocab_size=16000) as trainer:
  trainer.load_corpus("data.txt")
  trainer.train()
  trainer.save("model.bin", "vocab.txt")
```

### Complete Example

```python
from shredword.trainer import BPETrainer

trainer = BPETrainer(
  vocab_size=32000,
  unk_id=0,
  character_coverage=0.9995,
  min_pair_freq=2000
)

trainer.load_corpus("training_corpus.txt")
merges = trainer.train()
trainer.save("bpe_model.model", "bpe_vocab.vocab")
trainer.destroy()
```

### Multiple Corpus Training

```python
trainer = BPETrainer(vocab_size=25000)

corpus_files = ["corpus1.txt", "corpus2.txt", "corpus3.txt"]
for corpus_file in corpus_files:
  trainer.load_corpus(corpus_file)

trainer.train()
trainer.save("multi_corpus.model", "multi_corpus.vocab")
trainer.destroy()
```

### Error Handling

```python
try:
  trainer = BPETrainer(vocab_size=10000)
  trainer.load_corpus("corpus.txt")
  trainer.train()
  trainer.save("model.model", "vocab.vocab")
except RuntimeError as e:
  print(f"Training error: {e}")
except IOError as e:
  print(f"File error: {e}")
finally:
  trainer.destroy()
```

## C/C++ CLI

### Compilation

**Windows:**
```bash
g++ -o trainer.exe trainer.cpp bpe/bpe.cpp bpe/histogram.cpp bpe/hash.cpp bpe/heap.cpp unigram/unigram.cpp unigram/heap.cpp unigram/cache.cpp unigram/hashmap.cpp unigram/subword.cpp trie.cpp -I. -std=c++11
```

**Linux:**
```bash
g++ -o trainer.exe trainer.cpp bpe/bpe.cpp bpe/histogram.cpp bpe/hash.cpp bpe/heap.cpp unigram/unigram.cpp unigram/heap.cpp unigram/cache.cpp unigram/hashmap.cpp unigram/subword.cpp trie.cpp
```

### Usage

```bash
trainer.exe input=<corpus> model_type=bpe output_model=<model> output_vocab=<vocab> [options]
```

### Required Arguments

- `input=<path>`: Input corpus file path
- `model_type=bpe`: Specify BPE model type
- `output_model=<path>`: Output model file path
- `output_vocab=<path>`: Output vocabulary file path

### Optional Arguments

- `vocab_size=<int>`: Target vocabulary size (default: 32000)
- `character_coverage=<float>`: Character coverage 0.0-1.0 (default: 0.9995)
- `min_pair_freq=<int>`: Minimum pair frequency for merging (default: 2000)

### Examples

**Basic Training:**
```bash
trainer.exe input=corpus.txt model_type=bpe output_model=model.bin output_vocab=vocab.txt
```

**Custom Configuration:**
```bash
trainer.exe input=corpus.txt model_type=bpe output_model=model.bin output_vocab=vocab.txt vocab_size=50000 character_coverage=0.999 min_pair_freq=1000
```

### Training Process

The CLI performs three main steps:

1. **Corpus Loading**: Reads and processes input corpus file
2. **BPE Training**: Iteratively merges frequent character pairs
3. **Model Saving**: Outputs trained model and vocabulary files

### Output

```
========== BPE Training ==========
[CONFIG] Vocab Size: 32000
[CONFIG] Character Coverage: 0.9995
[CONFIG] Min Pair Freq: 2000

[STEP 1] Loading corpus from: corpus.txt
[INFO] Corpus loaded successfully. Vocabulary: 45231 words

[STEP 2] Training BPE model...
[SUCCESS] Training completed with 31999 merges

[STEP 3] Saving model and vocabulary...
[SUCCESS] Saved to:
  Model: model.bin
  Vocab: vocab.txt

========== Training Complete ==========
```

## Configuration Parameters

### vocab_size
Controls final vocabulary size. Larger vocabularies capture more specific subwords but increase model size.

**Typical Range:** 1,000 - 50,000

**Recommended values:**
- Small models: 8,192 - 16,384
- Medium models: 32,000 - 50,000
- Large models: 50,000 - 100,000

### unk_id
The ID assigned to out-of-vocabulary tokens during tokenization.

**Default:** 0

### character_coverage
Percentage of characters to cover from corpus. Higher values include rare characters.

**Range:** 0.0 - 1.0

**Recommended values:**
- English: 0.995 - 0.999
- Multilingual: 0.9995 - 1.0

### min_pair_freq
Minimum frequency threshold for merging pairs. Higher values result in more conservative merging.

**Recommended values:**
- Small corpus: 100 - 1,000
- Large corpus: 2,000 - 10,000

## File Format Requirements

### Corpus Format
- Plain text files
- UTF-8 encoding recommended
- One sentence per line (typical)
- No special preprocessing required

### Output Files
- **Model file (.model/.bin):** Contains the trained BPE merge operations
- **Vocabulary file (.vocab/.txt):** Contains the vocabulary mapping

## Best Practices

1. **Resource Management**: Always call `destroy()` or use context managers in Python
2. **Corpus Size**: Ensure corpus is large enough (typically millions of tokens) for meaningful training
3. **Corpus Quality**: Use diverse, representative text data
4. **Vocabulary Size**: Balance between granularity and memory
5. **Coverage**: Adjust based on language diversity
6. **Frequency Threshold**: Set based on corpus size and noise level
7. **File Paths**: Use absolute paths to avoid issues with relative path resolution
8. **Memory Usage**: Monitor memory usage with large corpora and adjust parameters accordingly

## Troubleshooting

### Common Issues

**"Failed to create BPE trainer"**
- Check that the underlying C library is properly installed
- Verify that configuration parameters are within valid ranges

**"Failed to load corpus"**
- Ensure the corpus file exists and is readable
- Check file encoding (UTF-8 is typically expected)
- Verify sufficient disk space and memory

**"Training failed"**
- Corpus may be too small or empty
- Try reducing `min_pair_freq` for small corpora
- Check available memory for large vocabularies

### Performance Tips

- Use SSD storage for faster corpus loading
- Consider the trade-off between vocabulary size and training time
- Monitor memory usage during training with large corpora
- For very large corpora, consider preprocessing to remove extremely rare characters
