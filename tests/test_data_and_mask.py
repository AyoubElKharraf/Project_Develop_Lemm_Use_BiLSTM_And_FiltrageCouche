import json
import pytest
from pathlib import Path
from src.arabic_utils import normalize_arabic, remove_tashkeel
from src.preprocess import parse_conllu_file

def test_normalization():
    # Test suppression tashkeel
    assert remove_tashkeel("ذَهَبَ") == "ذهب"
    # Test normalisation des Alifs
    assert normalize_arabic("أحمد") == "احمد"
    assert normalize_arabic("إلى") == "الى"
    assert normalize_arabic("آية") == "اية"

def test_conllu_mwt_filtering(tmp_path):
    # Création d'un mini-fichier CoNLL-U factice avec MWT
    conllu_content = """# visual-style 1 2 flat color:blue
1-2\tفبالله\t_\t_\t_\t_\t_\t_\t_\t_
1\tف\tفَ\tCCONJ\t_\t_\t_\t_\t_\t_
2\tبالله\tاللّٰه\tPROPN\t_\t_\t_\t_\t_\t_
3\tنستعين\tاِستَعَانَ\tVERB\t_\t_\t_\t_\t_\t_

"""
    dummy_file = tmp_path / "sample.conllu"
    dummy_file.write_text(conllu_content, encoding="utf-8")

    sentences = parse_conllu_file(dummy_file)
    assert len(sentences) == 1
    # Doit ignorer "1-2" et ne garder que 1, 2, 3
    assert len(sentences[0]["tokens_norm"]) == 3
    assert sentences[0]["tokens_norm"] == ["ف", "بالله", "نستعين"]
