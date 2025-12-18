# Unigram Language Model Tokenizer

## Overview

Unigram is a subword tokenization algorithm based on language modeling. It starts with a large seed vocabulary and iteratively prunes tokens with lowest likelihood scores, resulting in an optimal subword vocabulary that maximizes the likelihood of the training corpus.

## Installation

Ensure you have the required dependencies installed and the underlying C library (`cbase`) is properly configured.

From [PyPI.org](https://pypi.org/project/shredword-trainer/):

```bash
pip install shredword-trainer
```

Importing:

```python
from shredword.trainer import UnigramTrainer
```

## Python API

### Quick Start

```python
trainer = UnigramTrainer()
trainer.load_corpus("path/to/your/corpus.txt")
trainer.train()
trainer.save("base.vocab")
trainer.destroy()
```

### UnigramTrainer Class

#### Constructor

```python
UnigramTrainer(vocab_size=32000, character_coverage=0.9995, max_sentencepiece_length=16, seed_size=1000000)
```

**Parameters:**
- `vocab_size` (int): Target vocabulary size. Default: 32000
- `character_coverage` (float): Character coverage ratio (0.0-1.0). Default: 0.9995
- `max_sentencepiece_length` (int): Maximum length of sentence pieces. Default: 16
- `seed_size` (int): Initial seed vocabulary size. Default: 1000000

**Raises:**
- `RuntimeError`: If the trainer fails to initialize

#### Methods

##### load_corpus(path: str)

Loads training corpus from a text file into memory.

**Parameters:**
- `path` (str): Path to corpus file

**Raises:**
- `IOError`: If file doesn't exist

**Example:**
```python
trainer = UnigramTrainer(vocab_size=32000)
trainer.load_corpus("corpus.txt")
```

##### train(num_iterations: int = 10) -> int

Trains the Unigram model using EM algorithm.

**Parameters:**
- `num_iterations` (int): Number of EM iterations. Default: 10

**Returns:**
- `int`: Number of iterations performed

**Raises:**
- `RuntimeError`: If no texts loaded or training fails

**Example:**
```python
iterations = trainer.train(num_iterations=15)
print(f"Completed {iterations} iterations")
```

##### save(vocab_path: str)

Saves trained vocabulary to file.

**Parameters:**
- `vocab_path` (str): Output path for vocabulary file

**Raises:**
- `RuntimeError`: If saving fails

**Example:**
```python
trainer.save("vocab.txt")
```

##### destroy()

Releases trainer resources. Called automatically on context exit or deletion.

#### Context Manager Support

```python
with UnigramTrainer(vocab_size=50000) as trainer:
  trainer.load_corpus("data.txt")
  trainer.train(num_iterations=12)
  trainer.save("vocab.txt")
```

### Complete Example

```python
from shredword.trainer import UnigramTrainer

trainer = UnigramTrainer(
  vocab_size=50000,
  character_coverage=0.9995,
  max_sentencepiece_length=20,
  seed_size=2000000
)

trainer.load_corpus("training_corpus.txt")
trainer.train(num_iterations=10)
trainer.save("unigram_vocab.vocab")
trainer.destroy()
```

### Multiple Corpus Training

```python
trainer = UnigramTrainer(vocab_size=25000)

corpus_files = ["corpus1.txt", "corpus2.txt", "corpus3.txt"]
for corpus_file in corpus_files:
  trainer.load_corpus(corpus_file)

trainer.train(num_iterations=12)
trainer.save("multi_corpus.vocab")
trainer.destroy()
```

### Error Handling

```python
try:
  trainer = UnigramTrainer(vocab_size=10000)
  trainer.load_corpus("corpus.txt")
  trainer.train(num_iterations=10)
  trainer.save("vocab.vocab")
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
trainer.exe input=<corpus> model_type=unigram output_model=<model> output_vocab=<vocab> [options]
```

### Required Arguments

- `input=<path>`: Input corpus file path
- `model_type=unigram`: Specify Unigram model type
- `output_model=<path>`: Output model file path
- `output_vocab=<path>`: Output vocabulary file path

### Optional Arguments

- `vocab_size=<int>`: Target vocabulary size (default: 32000)
- `character_coverage=<float>`: Character coverage 0.0-1.0 (default: 0.9995)
- `max_piece_length=<int>`: Maximum sentence piece length (default: 16)
- `num_iterations=<int>`: Number of EM iterations (default: 10)
- `seed_size=<int>`: Initial seed vocabulary size (default: 1000000)

### Examples

**Basic Training:**
```bash
trainer.exe input=corpus.txt model_type=unigram output_model=model.bin output_vocab=vocab.txt
```

**Custom Configuration:**
```bash
trainer.exe input=corpus.txt model_type=unigram output_model=model.bin output_vocab=vocab.txt vocab_size=50000 character_coverage=0.999 num_iterations=15 max_piece_length=20
```

### Training Process

The CLI performs three main steps:

1. **Corpus Loading**: Reads and processes input corpus file (max texts limited)
2. **Unigram Training**: Iteratively optimizes vocabulary using EM algorithm
3. **Model Saving**: Outputs vocabulary file and model metadata

### Output

```
========== Unigram Training ==========
[CONFIG] Vocab Size: 32000
[CONFIG] Character Coverage: 0.9995
[CONFIG] Max Piece Length: 16
[CONFIG] Iterations: 10

[STEP 1] Loading corpus from: corpus.txt
[INFO] Loaded 125000 texts from corpus

[STEP 2] Training Unigram model...

[STEP 3] Saving vocabulary...
[SUCCESS] Saved vocabulary to: vocab.txt
[SUCCESS] Saved model metadata to: model.bin

========== Training Complete ==========
```

## Configuration Parameters

### vocab_size
Final vocabulary size after pruning. Determines tokenization granularity.

**Typical Range:** 1,000 - 128,000

**Recommended values:**
- Small models: 16,000 - 32,000
- Medium models: 32,000 - 64,000
- Large models: 64,000 - 128,000

### character_coverage
Percentage of characters to include in vocabulary. Critical for multilingual models.

**Range:** 0.0 - 1.0

**Recommended values:**
- Single language: 0.995 - 0.999
- Multilingual: 0.9995 - 1.0
- Mixed scripts: 0.9999 - 1.0

### max_sentencepiece_length
Maximum length of individual subword units in characters.

**Recommended values:**
- Agglutinative languages: 20 - 32
- Isolating languages: 12 - 16
- Mixed: 16 - 20

### seed_size
Initial vocabulary size before pruning. Larger seeds provide better coverage.

**Recommended values:**
- Small corpus: 500,000 - 1,000,000
- Large corpus: 1,000,000 - 5,000,000

### num_iterations
Number of Expectation-Maximization iterations. More iterations improve convergence.

**Recommended values:**
- Quick training: 5 - 8
- Standard training: 10 - 15
- High quality: 15 - 20

## File Format Requirements

### Corpus Format
- Plain text files
- UTF-8 encoding recommended
- One sentence per line (typical)
- No special preprocessing required

### Output Files
- **Model file (.model/.bin):** Contains model metadata
- **Vocabulary file (.vocab/.txt):** Contains the vocabulary mapping with scores

## Algorithm Details

### Training Process

1. **Seed Generation**: Creates large initial vocabulary from all possible substrings
2. **EM Iterations**: 
   - **E-step**: Computes likelihood of each subword in corpus
   - **M-step**: Updates subword probabilities
3. **Pruning**: Removes lowest-scoring subwords until target vocabulary size reached

### Advantages over BPE

- **Probabilistic**: Provides likelihood scores for each token
- **Multiple Segmentations**: Can generate multiple valid tokenizations
- **Better for Rare Words**: Handles infrequent words more effectively
- **Reversible**: Can decode tokens back to original text

## Best Practices

1. **Resource Management**: Always call `destroy()` or use context managers in Python
2. **Corpus Size**: Use large, diverse corpus (10M+ tokens recommended)
3. **Vocabulary Size**: Balance between granularity and model efficiency
4. **Iterations**: More iterations improve quality but increase training time
5. **Seed Size**: Set 20-50x larger than target vocabulary
6. **Max Length**: Adjust based on language morphology
7. **Character Coverage**: Set to 0.9995+ for production systems
8. **File Paths**: Use absolute paths to avoid issues with relative path resolution
9. **Memory Consideration**: Larger seed sizes require more RAM during training

## Troubleshooting

### Common Issues

**"Failed to create Unigram trainer"**
- Check that the underlying C library is properly installed
- Verify that configuration parameters are within valid ranges

**"No texts loaded"**
- Ensure the corpus file exists and is readable
- Check file encoding (UTF-8 is typically expected)
- Verify corpus is not empty

**"Training failed"**
- Corpus may be too small
- Try reducing `seed_size` or `vocab_size`
- Check available memory for large vocabularies
- Ensure sufficient iterations for convergence

### Performance Tips

- Use SSD storage for faster corpus loading
- Consider the trade-off between vocabulary size and training time
- Monitor memory usage during training with large corpora
- For very large corpora, consider preprocessing to remove extremely rare characters
- Adjust `seed_size` based on available memory

## Performance Tuning

### For Speed
- Reduce `num_iterations` to 5-8
- Decrease `seed_size` to 500,000
- Lower `max_sentencepiece_length` to 12

### For Quality
- Increase `num_iterations` to 15-20
- Raise `seed_size` to 2,000,000+
- Use `character_coverage` of 0.9999+
- Allow longer pieces with `max_sentencepiece_length` of 20-24

## Comparison with BPE

| Feature | Unigram | BPE |
|---------|---------|-----|
| Algorithm | EM-based pruning | Greedy merging |
| Probabilities | Yes | No |
| Multiple segmentations | Yes | No |
| Training speed | Slower | Faster |
| Memory usage | Higher | Lower |
| Quality | Better for rare words | Better for common words |
| Best for | Multilingual, morphologically rich | English, simple morphology |
