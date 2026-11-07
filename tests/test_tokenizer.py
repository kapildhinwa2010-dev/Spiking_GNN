import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import WordTokenizer


def test_roundtrip():
    tok = WordTokenizer(["the", "king", "queen"])
    ids = tok.encode("The KING!")
    assert ids == [0, 1]
    assert tok.decode(ids) == "the king"


def test_unknown_word():
    tok = WordTokenizer(["the", "king"])
    ids = tok.encode("the xyzzy")
    assert ids[1] == tok.unk_id
    assert "<unk>" in tok.decode(ids)


def test_from_text():
    tok = WordTokenizer.from_text("a a a b b c", vocab_size=2)
    assert tok.vocab == ["a", "b"]


def test_build_text_graph():
    tok = WordTokenizer(["a", "b", "c"])
    src, dst = tok.build_text_graph("a b c a")
    assert len(src) == 5
    assert all(0 <= s < 3 for s in src)
    assert all(0 <= d < 3 for d in dst)


def test_save_load(tmp_path):
    path = str(tmp_path / "vocab.json")
    tok = WordTokenizer(["a", "b"])
    tok.save(path)
    tok2 = WordTokenizer.load(path)
    assert tok2.vocab == ["a", "b"]
    assert tok2.encode("b a") == [1, 0]
