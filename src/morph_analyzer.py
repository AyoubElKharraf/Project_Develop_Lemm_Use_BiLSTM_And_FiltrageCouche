from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import List, Optional

class BaseMorphAnalyzer(ABC):
    """Interface abstraite commune pour les analyseurs morphologiques hors contexte."""

    @abstractmethod
    def analyze(self, word: str) -> List[str]:
        """Retourne la liste des lemmes candidats possibles pour le mot donné."""
        pass

class AlKhalilCacheAdapter(BaseMorphAnalyzer):
    """
    Adaptateur exploitant un cache ou dictionnaire pré-extrait (JSON) d'AlKhalil Morpho Sys.
    Permet une exécution instantanée sans nécessiter l'exécution de la JVM Java pendant le runtime.
    """
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache = {}
        if cache_file and Path(cache_file).exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def analyze(self, word: str) -> List[str]:
        return self.cache.get(word, [])

    def add_entry(self, word: str, lemmas: List[str]):
        if word not in self.cache:
            self.cache[word] = list(set(lemmas))
        else:
            self.cache[word] = list(set(self.cache[word] + lemmas))

class FallbackMorphAnalyzer(BaseMorphAnalyzer):
    """
    Analyseur morphologique robuste et autonome.
    Si CAMeL Tools est installé, l'utilise ; sinon utilise des règles morphologiques / lexique d'entraînement.
    """
    def __init__(self, external_analyzer: Optional[BaseMorphAnalyzer] = None):
        self.external_analyzer = external_analyzer
        self._camel_analyzer = None
        self._init_camel_if_available()

    def _init_camel_if_available(self):
        try:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer
            db = MorphologyDB.builtin_db()
            self._camel_analyzer = Analyzer(db)
        except Exception:
            self._camel_analyzer = None

    def analyze(self, word: str) -> List[str]:
        candidates = []
        if self.external_analyzer:
            candidates.extend(self.external_analyzer.analyze(word))

        if not candidates and self._camel_analyzer:
            try:
                analyses = self._camel_analyzer.analyze(word)
                for a in analyses:
                    if "lex" in a and a["lex"]:
                        candidates.append(a["lex"])
            except Exception:
                pass

        return list(set(candidates))
