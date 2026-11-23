import json
import logging
from collections import Counter
from pathlib import Path
from src.config import load_config
from src.morph_analyzer import FallbackMorphAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_vocabs_and_lmor():
    cfg = load_config()
    train_file = cfg.paths.processed_dir / "train.jsonl"
    artifacts_dir = cfg.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists():
        logger.error(f"Fichier d'entraînement manquant : {train_file}")
        return

    word_counts = Counter()
    lemma_counts = Counter()
    train_word2lemmas = {}

    logger.info("Extraction des statistiques depuis train.jsonl...")
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            tokens = item.get("tokens_norm", [])
            lemmas = item.get("lemmas", [])
            for w, l in zip(tokens, lemmas):
                word_counts[w] += 1
                lemma_counts[l] += 1
                if w not in train_word2lemmas:
                    train_word2lemmas[w] = set()
                train_word2lemmas[w].add(l)

    # Construction du vocabulaire de mots V_m
    pad_token = cfg.data.pad_token
    unk_token = cfg.data.unk_token
    pad_lemma = cfg.data.pad_lemma
    unk_lemma = cfg.data.unk_lemma

    word_token2id = {pad_token: cfg.data.pad_token_id, unk_token: cfg.data.unk_token_id}
    word_id2token = [pad_token, unk_token]
    for w, _ in word_counts.most_common():
        if w not in word_token2id:
            word_token2id[w] = len(word_id2token)
            word_id2token.append(w)

    # Construction du vocabulaire de lemmes V_l
    lemma_token2id = {pad_lemma: cfg.data.pad_lemma_id, unk_lemma: cfg.data.unk_lemma_id}
    lemma_id2token = [pad_lemma, unk_lemma]
    for l, _ in lemma_counts.most_common():
        if l not in lemma_token2id:
            lemma_token2id[l] = len(lemma_id2token)
            lemma_id2token.append(l)

    logger.info(f"Vocabulaire mots |V_m|: {len(word_id2token)}")
    logger.info(f"Vocabulaire lemmes |V_l|: {len(lemma_id2token)}")

    # Sauvegarde des vocabulaires
    with open(artifacts_dir / "vocab_words.json", "w", encoding="utf-8") as f:
        json.dump({"token2id": word_token2id, "id2token": word_id2token}, f, ensure_ascii=False, indent=2)

    with open(artifacts_dir / "vocab_lemmas.json", "w", encoding="utf-8") as f:
        json.dump({"token2id": lemma_token2id, "id2token": lemma_id2token}, f, ensure_ascii=False, indent=2)

    # Construction de L_mor
    # Fusion des candidats issus de l'analyseur morphologique et des paires du train
    analyzer = FallbackMorphAnalyzer()
    lmor = {}

    logger.info("Construction de L_mor (lexique morphologique mot -> IDs candidats)...")
    for word in word_token2id.keys():
        if word in [pad_token, unk_token]:
            continue

        candidate_lemmas = set(analyzer.analyze(word))
        # Règle de garantie : fusionner avec les paires observées dans le train uniquement
        if word in train_word2lemmas:
            candidate_lemmas.update(train_word2lemmas[word])

        # Convertir les lemmes textuels en IDs de V_l
        valid_lemma_ids = [
            lemma_token2id[lem] for lem in candidate_lemmas if lem in lemma_token2id
        ]

        if valid_lemma_ids:
            lmor[word] = sorted(list(set(valid_lemma_ids)))

    with open(artifacts_dir / "lmor.json", "w", encoding="utf-8") as f:
        json.dump(lmor, f, ensure_ascii=False, indent=2)

    logger.info(f"L_mor construit avec {len(lmor)} entrées.")

if __name__ == "__main__":
    build_vocabs_and_lmor()
