# ============================================================
# NEUROSPIKING WORD TOKENIZER
# Word-level tokenizer, consistent with train.py's vocab.json:
#   - build from a corpus (top-N most common words)
#   - save / load vocab.json
#   - encode text -> node ids (unknown words -> <unk>)
#   - decode ids -> text
#   - build_text_graph(): text -> co-occurrence graph edges
# ============================================================

import json
import re
from collections import Counter

UNK = "<unk>"


class WordTokenizer:
    def __init__(self, vocab):
        self.vocab = [w for w in vocab if w != UNK]
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.unk_id = len(self.vocab)

    # ---------- construction ----------
    @staticmethod
    def _tokenize(text):
        return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()

    @classmethod
    def from_text(cls, text, vocab_size=3000):
        tokens = cls._tokenize(text)
        vocab = [w for w, _ in Counter(tokens).most_common(vocab_size)]
        return cls(vocab)

    @classmethod
    def load(cls, path="vocab.json"):
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)

    def save(self, path="vocab.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)

    # ---------- encode / decode ----------
    def encode(self, text):
        return [self.word_to_id.get(w, self.unk_id) for w in self._tokenize(text)]

    def decode(self, ids):
        id_to_word = {i: w for w, i in self.word_to_id.items()}
        return " ".join(id_to_word.get(i, UNK) for i in ids)

    def __len__(self):
        return len(self.vocab) + 1  # + <unk>

    # ---------- text -> graph ----------
    def build_text_graph(self, text, window=5, max_edges=None):
        """Text -> co-occurrence edges (same recipe as train.py)."""
        ids = self.encode(text)
        vocab_size = len(self.vocab)
        src, dst = [], []
        for i in range(len(ids) - 1):
            if ids[i] >= vocab_size:
                continue
            for j in range(1, window + 1):
                if i + j >= len(ids):
                    break
                if ids[i + j] >= vocab_size:
                    continue
                if ids[i] != ids[i + j]:
                    src.append(ids[i])
                    dst.append(ids[i + j])
                if max_edges is not None and len(src) >= max_edges:
                    break
            if max_edges is not None and len(src) >= max_edges:
                break
        return src, dst
