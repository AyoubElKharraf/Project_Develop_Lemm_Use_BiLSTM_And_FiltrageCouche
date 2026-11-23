import argparse
import json
from pathlib import Path
import torch
from src.config import load_config
from src.arabic_utils import normalize_arabic
from src.models import BiLSTMFiltered, BiLSTMBaseline

def lemmatize_sentence(sentence: str, model_type: str = "filtered"):
    cfg = load_config()
    device = torch.device(cfg.training.device)

    with open(cfg.paths.artifacts_dir / "vocab_words.json", "r", encoding="utf-8") as f:
        word2id = json.load(f)["token2id"]

    with open(cfg.paths.artifacts_dir / "vocab_lemmas.json", "r", encoding="utf-8") as f:
        lem_data = json.load(f)
        lemma2id = lem_data["token2id"]
        id2lemma = lem_data["id2token"]

    with open(cfg.paths.artifacts_dir / "lmor.json", "r", encoding="utf-8") as f:
        lmor = json.load(f)

    # Tokenisation simple sur les espaces
    raw_tokens = sentence.strip().split()
    tokens_norm = [normalize_arabic(w) for w in raw_tokens]

    unk_w_id = word2id.get("<UNK>", 1)
    word_ids = [word2id.get(w, unk_w_id) for w in tokens_norm]

    seq_len = len(word_ids)
    num_lemmas = len(id2lemma)

    word_ids_tensor = torch.tensor([word_ids], dtype=torch.long, device=device)
    mask_tensor = torch.zeros((1, seq_len, num_lemmas), dtype=torch.bool, device=device)

    for t, w in enumerate(tokens_norm):
        cands = lmor.get(w, [])
        if not cands:
            mask_tensor[0, t, :] = True  # fallback anti-NaN
        else:
            mask_tensor[0, t, cands] = True

    # Chargement du modèle
    ckpt_path = cfg.paths.checkpoints_dir / f"{model_type}_best.pt"
    if not ckpt_path.exists():
        print(f"Erreur : Modèle {ckpt_path} introuvable.")
        return

    is_filtered = (model_type == "filtered")
    if is_filtered:
        model = BiLSTMFiltered(len(word2id), num_lemmas, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.model.dropout)
    else:
        model = BiLSTMBaseline(len(word2id), num_lemmas, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.model.dropout)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        if is_filtered:
            logits = model(word_ids_tensor, mask=mask_tensor)
        else:
            logits = model(word_ids_tensor)

        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    print(f"\nRésultat de lemmatisation ({model_type.upper()}) :")
    print(f"{'Mot Original':<20} | {'Mot Normalisé':<20} | {'Lemme Prédit':<20} | {'Confiance':<10}")
    print("-" * 75)
    for orig, norm, p_id, p_tensor in zip(raw_tokens, tokens_norm, pred_ids, probs[0]):
        pred_lem = id2lemma[p_id] if p_id < len(id2lemma) else "<UNK>"
        conf = p_tensor[p_id].item() * 100
        print(f"{orig:<20} | {norm:<20} | {pred_lem:<20} | {conf:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Inférence CLI du lemmatiseur BiLSTM arabe")
    parser.add_argument("--text", type=str, default="ذهب الطالب إلى المدرسة", help="Phrase arabe à lemmatiser")
    parser.add_argument("--model", type=str, choices=["baseline", "filtered"], default="filtered")
    args = parser.parse_args()

    lemmatize_sentence(args.text, args.model)

if __name__ == "__main__":
    main()
