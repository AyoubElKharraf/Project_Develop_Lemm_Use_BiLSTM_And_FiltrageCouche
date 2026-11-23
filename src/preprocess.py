import json
import logging
from pathlib import Path
from src.config import load_config
from src.arabic_utils import normalize_arabic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_conllu_file(conllu_path: Path) -> list[dict]:
    """
    Parse un fichier CoNLL-U en filtrant strictement les tokens multi-mots (MWT).
    On ne garde que les tokens avec id entier (isinstance(id, int)).
    Extrait 'tokens_norm' (dé-diacritisés et normalisés) et 'lemmas' (vocalisés tels quels).
    """
    sentences = []
    current_tokens = []
    current_lemmas = []

    with open(conllu_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append({
                        "tokens_norm": current_tokens,
                        "lemmas": current_lemmas
                    })
                    current_tokens = []
                    current_lemmas = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                token_id_str = parts[0]
                # Filtrer les multi-word tokens comme "4-5" ou les nœuds vides comme "4.1"
                if not token_id_str.isdigit():
                    continue

                form = parts[1]
                lemma = parts[2]

                # Nettoyage / normalisation
                norm_form = normalize_arabic(form)
                # Si vide après normalisation, conserver la forme brute épurée
                if not norm_form:
                    norm_form = form.strip()

                current_tokens.append(norm_form)
                current_lemmas.append(lemma.strip())

    if current_tokens:
        sentences.append({
            "tokens_norm": current_tokens,
            "lemmas": current_lemmas
        })

    return sentences

def save_jsonl(data: list[dict], out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Sauvegardé {len(data)} phrases dans {out_path}")

def main():
    cfg = load_config()
    raw_dir = cfg.paths.raw_dir
    processed_dir = cfg.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    file_mapping = {
        "train": "ar_padt-ud-train.conllu",
        "dev": "ar_padt-ud-dev.conllu",
        "test": "ar_padt-ud-test.conllu"
    }

    for split, filename in file_mapping.items():
        src_file = raw_dir / filename
        out_file = processed_dir / f"{split}.jsonl"
        if not src_file.exists():
            logger.warning(f"Fichier source non trouvé : {src_file}. Lancez src/download_data.py d'abord.")
            continue
        logger.info(f"Prétraitement de {src_file.name}...")
        sentences = parse_conllu_file(src_file)
        save_jsonl(sentences, out_file)

if __name__ == "__main__":
    main()
