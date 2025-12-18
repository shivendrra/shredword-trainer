import ctypes, os, sys, platform, sysconfig
from ctypes import Structure, c_float, c_int, c_int32, c_uint64, c_size_t, c_char_p, POINTER, c_bool, c_double

def _get_lib_path():
  pkg_dir = os.path.dirname(__file__)
  possible_names, candidates = ['trainer', 'libtrainer'], []
  possible_exts = ['.pyd', '.dll', '.so', '.dylib', sysconfig.get_config_var('EXT_SUFFIX') or '']
  search_dirs = [pkg_dir, os.path.join(pkg_dir, 'lib'), os.path.join(pkg_dir, '..', 'build')]

  for search_dir in search_dirs:
    if not os.path.exists(search_dir): continue
    try:
      for root, dirs, files in os.walk(search_dir):
        for file in files:
          for name in possible_names:
            if file.startswith(name) and any(file.endswith(ext) for ext in possible_exts if ext): candidates.append(os.path.join(root, file))
    except OSError: continue

  if candidates: return os.path.abspath(candidates[0])
  available = []
  for d in search_dirs:
    if os.path.exists(d):
      try: available.extend(os.listdir(d))
      except OSError: pass

  raise FileNotFoundError(f"Could not find trainer library in {search_dirs}. Available files: {available}")

_lib_path = _get_lib_path()
if hasattr(ctypes, 'RTLD_GLOBAL'): lib = ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)
else: lib = ctypes.CDLL(_lib_path)

MIN_HEAP_SIZE, MAX_OCCS_PER_MERGE, INITIAL_VOCAB_SIZE, INITIAL_STR_SIZE = 4096, 50000, 256, 4096

class Symbol(Structure): pass
class WordPos(Structure): pass
class Corpus(Structure): pass
class BPEConfig(Structure): pass
class Trainer(Structure): pass
class MaxHeap(Structure): pass
class BIMap(Structure): pass
class PairKey(Structure): pass
class UnigramTrainer(Structure): pass

Symbol._fields_ = [("id", c_int32), ("prev", POINTER(Symbol)), ("next", POINTER(Symbol)), ("deleted", c_bool)]
WordPos._fields_ = [("word_index", c_size_t), ("pos", POINTER(Symbol))]
Corpus._fields_ = [("words", POINTER(POINTER(Symbol))), ("word_counts", POINTER(c_uint64)), ("vocab_size", c_size_t)]
BPEConfig._fields_ = [("target_vocab_size", c_size_t), ("unk_id", c_int32), ("character_coverage", c_float), ("min_pair_freq", c_uint64)]
Trainer._fields_ = [("config", BPEConfig), ("heap", POINTER(MaxHeap)), ("corpus", POINTER(Corpus)), ("bigram_map", POINTER(BIMap)), ("next_token", c_size_t), ("num_merges", c_size_t), ("merge_ops", POINTER(PairKey)), ("token_strs", POINTER(c_char_p)), ("token_freq", POINTER(c_uint64))]

lib.create_trainer.argtypes, lib.create_trainer.restype = [POINTER(BPEConfig)], POINTER(Trainer)
lib.bpe_trainer_destroy.argtypes, lib.bpe_trainer_destroy.restype = [POINTER(Trainer)], None
lib.bpe_init.argtypes, lib.bpe_init.restype = [POINTER(Trainer)], None
lib.bpe_count_bigrams.argtypes, lib.bpe_count_bigrams.restype = [POINTER(Trainer)], None
lib.bpe_load_corpus.argtypes, lib.bpe_load_corpus.restype = [POINTER(Trainer), c_char_p], c_int
lib.bpe_merge_batch.argtypes, lib.bpe_merge_batch.restype = [POINTER(Trainer), c_int], c_int
lib.bpe_train.argtypes, lib.bpe_train.restype = [POINTER(Trainer)], c_int
lib.bpe_save.argtypes, lib.bpe_save.restype = [POINTER(Trainer), c_char_p, c_char_p], None

lib.trainerCreate.argtypes, lib.trainerCreate.restype = [c_int, c_float, c_int, c_int], POINTER(UnigramTrainer)
lib.trainerDestroy.argtypes, lib.trainerDestroy.restype = [POINTER(UnigramTrainer)], None
lib.addTextToTrainer.argtypes, lib.addTextToTrainer.restype = [POINTER(UnigramTrainer), c_char_p], c_bool
lib.preprocessTexts.argtypes, lib.preprocessTexts.restype = [POINTER(UnigramTrainer)], c_bool
lib.extractInitialSubwords.argtypes, lib.extractInitialSubwords.restype = [POINTER(UnigramTrainer)], c_bool
lib.computeLoss.argtypes, lib.computeLoss.restype = [POINTER(UnigramTrainer), POINTER(c_char_p), c_int], c_float
lib.computeTokenLoss.argtypes, lib.computeTokenLoss.restype = [POINTER(UnigramTrainer), c_char_p, POINTER(c_char_p), c_int], c_double
lib.pruneVocabStep.argtypes, lib.pruneVocabStep.restype = [POINTER(UnigramTrainer), POINTER(c_char_p), c_int, c_double], c_bool
lib.updateTokenScores.argtypes, lib.updateTokenScores.restype = [POINTER(UnigramTrainer), POINTER(c_char_p), c_int], c_bool
lib.trainUnigram.argtypes, lib.trainUnigram.restype = [POINTER(UnigramTrainer), POINTER(c_char_p), c_int, c_int], c_bool
lib.getVocab.argtypes, lib.getVocab.restype = [POINTER(UnigramTrainer), POINTER(POINTER(c_char_p)), POINTER(POINTER(c_double)), POINTER(c_int)], c_bool
lib.saveVocab.argtypes, lib.saveVocab.restype = [POINTER(UnigramTrainer), c_char_p], c_bool
lib.loadVocab.argtypes, lib.loadVocab.restype = [POINTER(UnigramTrainer), c_char_p], c_bool