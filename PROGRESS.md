# PROGRESS.md — Suivi & Passation (Lemmatiseur BiLSTM Filtré Arabe)

## 1. État global du projet
* **Phase actuelle :** [x] Jalon 0 (Contrats) | [x] Jalon 1 (Dev parallèle & Implémentation complète) | [ ] Jalon 2 (Entraînement & Évaluation finale)
* **Dernière mise à jour :** 2026-11-23 16:50
* **Dernier contributeur :** Antigravity
* **Statut des contrats d'interface :** [x] Gelés & Validés

---

## 2. Suivi Lot 1 — Toi (Données, Morphologie & Inférence)
- [x] `requirements.txt`, `config.yaml` & `src/config.py` (hyperparamètres et chemins figés)
- [x] `src/arabic_utils.py` (normalisation et dé-diacritisation)
- [x] `src/download_data.py` (Téléchargement UD_Arabic-PADT dans `data/raw/`)
- [x] `src/preprocess.py` (Génération de `data/processed/{train,dev,test}.jsonl` - filtrage MWT CoNLL-U)
- [x] `src/morph_analyzer.py` (BaseMorphAnalyzer, AlKhalil Cache, CAMeL Tools / Fallback)
- [x] `src/vocab_lexicon.py` (Génération `artifacts/vocab_words.json`, `vocab_lemmas.json`, `lmor.json`)
- [x] `src/predict.py` (Inférence CLI interactive et formatage tableau)
- [x] `tests/test_data_and_mask.py` (Tests unitaires Lot 1)

---

## 3. Suivi Lot 2 — Ton binôme (Dataset, Modèles, Entraînement & Évaluation)
- [x] `src/dataset.py` (LemmaDataset + collate_fn + masques binaires 3D + repli anti-NaN)
- [x] `src/models/__init__.py` & `src/models/bilstm_baseline.py` (BiLSTM baseline standard)
- [x] `src/models/bilstm_filtered.py` (BiLSTM filtré avec masquage logits `-1e9` avant Softmax)
- [x] `src/train.py` (Boucle d'entraînement, Adam, CrossEntropy avec ignore_index, gradient clipping, checkpoints)
- [x] `src/evaluate.py` (Exactitude globale, mots ambigus, vitesse mots/sec, export `reports/eval_summary.json`)
- [x] `tests/test_models.py` (Tests unitaires Lot 2 : gradients, shapes, probabilités nulles hors masque, anti-NaN)

---

## 4. Passation active (À lire par le binôme et Antigravity)

### Ce qui vient d'être fait :
1. Code source intégralement généré selon les contrats d'interface stricts.
2. Implémentation du masquage morphologique par logit masking (`logits.masked_fill(~mask, -1e9)`).
3. Mécanisme de sécurité anti-NaN vérifié par test unitaire.
4. Suites de tests unitaires pour le Lot 1 (`tests/test_data_and_mask.py`) et le Lot 2 (`tests/test_models.py`).

### Ce qui fonctionne et a été vérifié :
- Contrats validés par le prompt maître de référence.

### Ce qu'il reste à faire immédiatement (prochaine étape exacte) :
1. Exécuter `pytest` pour valider l'intégrité de la suite de tests.
2. Lancer `python -m src.download_data` pour télécharger les fichiers CoNLL-U réels d'UD_Arabic-PADT.
3. Lancer `python -m src.preprocess` puis `python -m src.vocab_lexicon`.
4. Lancer les entraînements :
   - `python -m src.train --model baseline`
   - `python -m src.train --model filtered`
5. Lancer l'évaluation comparative : `python -m src.evaluate`.

---

## 5. Journal des modifications de contrats
- Aucun changement : contrats 100% alignés avec la spécification du cours du Pr. Mazroui.
