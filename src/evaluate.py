import argparse
import json
import logging
import time
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset import LemmaDataset, pad_collate_fn
from src.models import BiLSTMBaseline, BiLSTMFiltered

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_detailed(model, dataloader, device, is_filtered: bool, lmor_dict: dict, id2lemma: list):
    model.eval()
    total_tokens = 0
    correct_tokens = 0

    ambiguous_tokens = 0
    ambiguous_correct = 0

    oov_tokens = 0
    oov_correct = 0

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Évaluation Test"):
            word_ids = batch["word_ids"].to(device)
            lemma_ids = batch["lemma_ids"].to(device)
            mask = batch["candidate_mask"].to(device) if is_filtered else None

            if is_filtered:
                logits = model(word_ids, mask=mask)
            else:
                logits = model(word_ids)

            preds = torch.argmax(logits, dim=-1)

            # Analyse mot par mot
            for b in range(word_ids.size(0)):
                for t in range(batch["lengths"][b]):
                    target = lemma_ids[b, t].item()
                    pred = preds[b, t].item()

                    if target == 0:  # PAD
                        continue

                    total_tokens += 1
                    is_correct = (pred == target)
                    if is_correct:
                        correct_tokens += 1

                    # Récupération des candidats du mot
                    cands = batch["candidate_mask"][b, t].nonzero().flatten().tolist() if is_filtered else []

                    # Est-ce ambigu (plus de 1 lemme candidat dans L_mor) ?
                    if len(cands) > 1 and len(cands) < len(id2lemma):
                        ambiguous_tokens += 1
                        if is_correct:
                            ambiguous_correct += 1

    elapsed = time.time() - start_time
    words_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    accuracy = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
    ambig_accuracy = (ambiguous_correct / ambiguous_tokens) * 100 if ambiguous_tokens > 0 else 0.0

    return {
        "accuracy": accuracy,
        "error_rate": 100.0 - accuracy,
        "ambiguous_accuracy": ambig_accuracy,
        "words_per_sec": words_per_sec,
        "total_words": total_tokens,
        "ambiguous_words": ambiguous_tokens
    }

def main():
    cfg = load_config()
    device = torch.device(cfg.training.device)

    test_jsonl = cfg.paths.processed_dir / "test.jsonl"
    vocab_words_path = cfg.paths.artifacts_dir / "vocab_words.json"
    vocab_lemmas_path = cfg.paths.artifacts_dir / "vocab_lemmas.json"
    lmor_path = cfg.paths.artifacts_dir / "lmor.json"
    reports_dir = cfg.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = LemmaDataset(test_jsonl, vocab_words_path, vocab_lemmas_path, lmor_path, max_seq_len=cfg.data.max_seq_len)
    collate = partial(pad_collate_fn, num_lemmas=test_dataset.num_lemmas, pad_word_id=cfg.data.pad_token_id, pad_lemma_id=cfg.data.pad_lemma_id)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate)

    results = {}

    for model_name in ["baseline", "filtered"]:
        ckpt_path = cfg.paths.checkpoints_dir / f"{model_name}_best.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint {ckpt_path} introuvable. Passez le modèle {model_name}.")
            continue

        logger.info(f"Chargement et évaluation de {model_name}...")
        is_filtered = (model_name == "filtered")
        if is_filtered:
            model = BiLSTMFiltered(len(test_dataset.word2id), test_dataset.num_lemmas, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.model.dropout)
        else:
            model = BiLSTMBaseline(len(test_dataset.word2id), test_dataset.num_lemmas, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.model.dropout)

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        res = evaluate_detailed(model, test_loader, device, is_filtered, test_dataset.lmor, test_dataset.id2lemma)
        results[model_name] = res

        logger.info(f"[{model_name.upper()}] Exactitude: {res['accuracy']:.2f}% | Erreur: {res['error_rate']:.2f}% | Vitesse: {res['words_per_sec']:.1f} mots/s | Acc Ambigus: {res['ambiguous_accuracy']:.2f}%")

    out_file = reports_dir / "eval_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Rapport sauvegardé dans {out_file}")

if __name__ == "__main__":
    main()
