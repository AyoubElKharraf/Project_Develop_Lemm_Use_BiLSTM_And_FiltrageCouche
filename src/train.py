import argparse
import json
import logging
from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset import LemmaDataset, pad_collate_fn
from src.models import BiLSTMBaseline, BiLSTMFiltered

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, criterion, device, is_filtered: bool, grad_clip: float):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Entraînement", leave=False):
        word_ids = batch["word_ids"].to(device)
        lemma_ids = batch["lemma_ids"].to(device)
        mask = batch["candidate_mask"].to(device) if is_filtered else None

        optimizer.zero_grad()

        if is_filtered:
            logits = model(word_ids, mask=mask)
        else:
            logits = model(word_ids)

        # Reshape pour CrossEntropyLoss: (B * T, |V_l|) vs (B * T)
        loss = criterion(logits.view(-1, logits.size(-1)), lemma_ids.view(-1))
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

        # Calcul exactitude hors padding
        preds = torch.argmax(logits, dim=-1)
        valid_mask = (lemma_ids != 0)
        correct += ((preds == lemma_ids) & valid_mask).sum().item()
        total += valid_mask.sum().item()

    acc = (correct / total) * 100 if total > 0 else 0.0
    return total_loss / len(dataloader), acc

def evaluate(model, dataloader, criterion, device, is_filtered: bool):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            word_ids = batch["word_ids"].to(device)
            lemma_ids = batch["lemma_ids"].to(device)
            mask = batch["candidate_mask"].to(device) if is_filtered else None

            if is_filtered:
                logits = model(word_ids, mask=mask)
            else:
                logits = model(word_ids)

            loss = criterion(logits.view(-1, logits.size(-1)), lemma_ids.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            valid_mask = (lemma_ids != 0)
            correct += ((preds == lemma_ids) & valid_mask).sum().item()
            total += valid_mask.sum().item()

    acc = (correct / total) * 100 if total > 0 else 0.0
    return total_loss / len(dataloader), acc

def main():
    parser = argparse.ArgumentParser(description="Entraînement du lemmatiseur BiLSTM (Baseline ou Filtré)")
    parser.add_argument("--model", type=str, choices=["baseline", "filtered"], default="filtered",
                        help="Type de modèle à entraîner")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch_size", type=int, default=None, help="Taille des mini-lots")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device(cfg.training.device)
    epochs = args.epochs if args.epochs else cfg.training.epochs
    batch_size = args.batch_size if args.batch_size else cfg.training.batch_size

    train_jsonl = cfg.paths.processed_dir / "train.jsonl"
    dev_jsonl = cfg.paths.processed_dir / "dev.jsonl"
    vocab_words_path = cfg.paths.artifacts_dir / "vocab_words.json"
    vocab_lemmas_path = cfg.paths.artifacts_dir / "vocab_lemmas.json"
    lmor_path = cfg.paths.artifacts_dir / "lmor.json"

    train_dataset = LemmaDataset(train_jsonl, vocab_words_path, vocab_lemmas_path, lmor_path, max_seq_len=cfg.data.max_seq_len)
    dev_dataset = LemmaDataset(dev_jsonl, vocab_words_path, vocab_lemmas_path, lmor_path, max_seq_len=cfg.data.max_seq_len)

    collate = partial(pad_collate_fn, num_lemmas=train_dataset.num_lemmas, pad_word_id=cfg.data.pad_token_id, pad_lemma_id=cfg.data.pad_lemma_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    is_filtered = (args.model == "filtered")
    vocab_w_len = len(train_dataset.word2id)
    vocab_l_len = train_dataset.num_lemmas

    if is_filtered:
        logger.info(f"Instanciation de BiLSTM Filtré (|V_w|={vocab_w_len}, |V_l|={vocab_l_len})...")
        model = BiLSTMFiltered(vocab_w_len, vocab_l_len, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.model.dropout, pad_idx=cfg.data.pad_token_id)
    else:
        logger.info(f"Instanciation de BiLSTM Baseline (|V_w|={vocab_w_len}, |V_l|={vocab_l_len})...")
        model = BiLSTMBaseline(vocab_w_len, vocab_l_len, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.model.dropout, pad_idx=cfg.data.pad_token_id)

    model.to(device)

    # Perte avec ignore_index sur PAD
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.data.pad_lemma_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    best_dev_acc = 0.0
    checkpoint_path = cfg.paths.checkpoints_dir / f"{args.model}_best.pt"

    logger.info(f"Début de l'entraînement ({epochs} époques sur {device})...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, is_filtered, cfg.training.grad_clip)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device, is_filtered)

        logger.info(f"Époque {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} - Acc: {tr_acc:.2f}% | Dev Loss: {dev_loss:.4f} - Acc: {dev_acc:.2f}%")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_dev_acc,
                "model_type": args.model
            }, checkpoint_path)
            logger.info(f" -> Meilleur modèle sauvegardé : {checkpoint_path} (Acc: {best_dev_acc:.2f}%)")

if __name__ == "__main__":
    main()
