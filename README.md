# ShredWord

ShredWord is a high-performance tokenizer training library supporting Byte-Pair Encoding (BPE) and Unigram Language Model algorithms. Designed for fast, efficient, and flexible text processing, it provides vocabulary training functionalities backed by a C/C++ core with a Python interface for seamless integration into machine learning workflows.

## Features

1. **Multiple Tokenization Algorithms**: Supports both BPE and Unigram training methods for flexible vocabulary generation
2. **Efficient Tokenization**: Utilizes optimized algorithms for compressing text data and reducing vocabulary size
3. **Customizable Vocabulary**: Allows users to define target vocabulary size, character coverage, and algorithm-specific parameters
4. **Save and Load Models**: Supports saving and loading trained tokenizers for reuse across projects
5. **Python Integration**: Provides a clean Python interface for seamless integration into NLP pipelines
6. **C/C++ CLI**: Includes command-line interface for direct training without Python dependencies

## How It Works

### Byte-Pair Encoding (BPE)

BPE is a subword tokenization algorithm that compresses a dataset by iteratively merging the most frequent pairs of characters or subwords into new tokens. This process continues until a predefined vocabulary size is reached.

Key steps:

1. Initialize the vocabulary with all unique characters in the dataset
2. Count the frequency of character pairs
3. Merge the most frequent pair into a new token
4. Repeat until the target vocabulary size is achieved

### Unigram Language Model

Unigram is a probabilistic subword tokenization algorithm that starts with a large seed vocabulary and iteratively prunes tokens with the lowest likelihood scores using the Expectation-Maximization (EM) algorithm.

Key steps:

1. Generate a large initial vocabulary from all possible substrings
2. Compute likelihood scores for each subword in the corpus
3. Update subword probabilities using EM iterations
4. Prune lowest-scoring subwords until target vocabulary size is reached

ShredWord implements both algorithms efficiently in C/C++, exposing training and vocabulary management methods through Python.

## Installation

### Prerequisites

- Python 3.11+
- GCC or a compatible compiler (for building from source)

### Steps

Install the Python package from [PyPI.org](https://pypi.org/project/shredword-trainer/):

```bash
pip install shredword-trainer
```

## Usage

Below are examples demonstrating how to use ShredWord for training tokenizers with both BPE and Unigram algorithms.

### BPE Trainer

```python
from shredword.trainer import BPETrainer

trainer = BPETrainer(
  vocab_size=8192,
  unk_id=0,
  character_coverage=0.995,
  min_pair_freq=2000
)

trainer.load_corpus("data/corpus.txt")
trainer.train()
trainer.save("model/bpe.model", "model/bpe.vocab")
trainer.destroy()
```

### Unigram Trainer

Note: Unigram implementation is currently under development and may not be fully functional.

```python
from shredword.trainer import UnigramTrainer

trainer = UnigramTrainer(
  vocab_size=32000,
  character_coverage=0.9995,
  max_sentencepiece_length=16,
  seed_size=1000000
)

trainer.load_corpus("data/corpus.txt")
trainer.train(num_iterations=10)
trainer.save("model/unigram.vocab")
trainer.destroy()
```

### Context Manager Pattern

```python
from shredword.trainer import BPETrainer

with BPETrainer(vocab_size=16000) as trainer:
  trainer.load_corpus("data/corpus.txt")
  trainer.train()
  trainer.save("model/bpe.model", "model/bpe.vocab")
```

### Multiple Corpus Training

```python
from shredword.trainer import BPETrainer

trainer = BPETrainer(vocab_size=25000)

corpus_files = ["data/corpus1.txt", "data/corpus2.txt", "data/corpus3.txt"]
for corpus_file in corpus_files:
  trainer.load_corpus(corpus_file)

trainer.train()
trainer.save("model/multi.model", "model/multi.vocab")
trainer.destroy()
```

## API Overview

### BPETrainer

#### Constructor Parameters

- `vocab_size` (int): Target vocabulary size. Default: 8192
- `unk_id` (int): ID for unknown tokens. Default: 0
- `character_coverage` (float): Character coverage ratio (0.0-1.0). Default: 0.995
- `min_pair_freq` (int): Minimum frequency for pair merging. Default: 2000

#### Methods

- `load_corpus(path)`: Load training corpus from a text file
- `train()`: Train the BPE model on loaded corpus
- `save(model_path, vocab_path)`: Save trained model and vocabulary
- `destroy()`: Release trainer resources

### UnigramTrainer

#### Constructor Parameters

- `vocab_size` (int): Target vocabulary size. Default: 32000
- `character_coverage` (float): Character coverage ratio (0.0-1.0). Default: 0.9995
- `max_sentencepiece_length` (int): Maximum length of sentence pieces. Default: 16
- `seed_size` (int): Initial seed vocabulary size. Default: 1000000

#### Methods

- `load_corpus(path)`: Load training corpus from a text file
- `train(num_iterations)`: Train the Unigram model using EM algorithm
- `save(vocab_path)`: Save trained vocabulary
- `destroy()`: Release trainer resources

## C/C++ CLI Usage

ShredWord also provides a command-line interface for training directly without Python.

### Compilation

**Windows:**
```bash
g++ -o trainer.exe trainer.cpp bpe/bpe.cpp bpe/histogram.cpp bpe/hash.cpp bpe/heap.cpp unigram/unigram.cpp unigram/heap.cpp unigram/cache.cpp unigram/hashmap.cpp unigram/subword.cpp trie.cpp -I. -std=c++11
```

**Linux:**
```bash
g++ -o trainer.exe trainer.cpp bpe/bpe.cpp bpe/histogram.cpp bpe/hash.cpp bpe/heap.cpp unigram/unigram.cpp unigram/heap.cpp unigram/cache.cpp unigram/hashmap.cpp unigram/subword.cpp trie.cpp
```

### Training with CLI

**BPE:**
```bash
trainer.exe input=corpus.txt model_type=bpe output_model=model.bin output_vocab=vocab.txt vocab_size=32000
```

**Unigram:**
```bash
trainer.exe input=corpus.txt model_type=unigram output_model=model.bin output_vocab=vocab.txt vocab_size=32000 num_iterations=10
```

## Advanced Features

### Error Handling

```python
from shredword.trainer import BPETrainer

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

### Resource Management

Always call `destroy()` to properly clean up resources, or use the context manager pattern for automatic cleanup:

```python
with BPETrainer(vocab_size=16000) as trainer:
  trainer.load_corpus("data.txt")
  trainer.train()
  trainer.save("model.model", "vocab.vocab")
```

## Configuration Guidelines

### Vocabulary Size

- Small models: 8,192 - 16,384
- Medium models: 32,000 - 50,000
- Large models: 50,000 - 100,000

### Character Coverage

- English: 0.995 - 0.999
- Multilingual: 0.9995 - 1.0

### Minimum Pair Frequency (BPE)

- Small corpus: 100 - 1,000
- Large corpus: 2,000 - 10,000

### EM Iterations (Unigram)

- Quick training: 5 - 8
- Standard training: 10 - 15
- High quality: 15 - 20

## File Formats

### Input Corpus

- Plain text files
- UTF-8 encoding recommended
- One sentence per line (typical)

### Output Files

- **Model file (.model/.bin)**: Contains merge operations (BPE) or metadata (Unigram)
- **Vocabulary file (.vocab/.txt)**: Contains vocabulary mapping

## Documentation

For detailed documentation on both BPE and Unigram trainers, including API references, configuration parameters, and troubleshooting guides, refer to:

- [BPE Documentation](docs/bpe.md)
- [Unigram Documentation](docs/unigram.md)

## Known Limitations

- Unigram implementation is currently under development and may not function as expected
- Maximum corpus size may be limited by available system memory
- CLI interface has a maximum text limit for Unigram training

## Project Information

A project by Shivendra
