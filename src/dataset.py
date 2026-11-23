import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import List, Dict

class LemmaDataset(Dataset):
    """
    Dataset PyTorch pour la lemmatisation sous contexte avec génération de masques.
    Contrat :
      - Entrée : tokens_norm (IDs de mots)
      - Cible : lemmas (IDs de lemmes)
      - Masque morphologique par token : tenseur booléen (len_vocab_l) où True = candidat valide.
    """
    def __init__(self, jsonl_path: Path, vocab_words_path: Path, vocab_lemmas_path: Path, lmor_path: Path, max_seq_len: int = 57):
        self.max_seq_len = max_seq_len

        with open(vocab_words_path, "r", encoding="utf-8") as f:
            self.word2id = json.load(f)["token2id"]

        with open(vocab_lemmas_path, "r", encoding="utf-8") as f:
            vocab_lem = json.load(f)
            self.lemma2id = vocab_lem["token2id"]
            self.id2lemma = vocab_lem["id2token"]

        self.num_lemmas = len(self.id2lemma)

        with open(lmor_path, "r", encoding="utf-8") as f:
            self.lmor = json.load(f)  # dict : mot -> list[int] (lemma IDs)

        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("tokens_norm"):
                    self.samples.append(item)

        self.unk_word_id = self.word2id.get("<UNK>", 1)
        self.unk_lemma_id = self.lemma2id.get("<UNK>", 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = item["tokens_norm"][:self.max_seq_len]
        lemmas = item["lemmas"][:self.max_seq_len]

        word_ids = [self.word2id.get(w, self.unk_word_id) for w in tokens]
        lemma_ids = [self.lemma2id.get(l, self.unk_lemma_id) for l in lemmas]

        # Récupération des candidats pour chaque mot
        cands_per_word = []
        for w in tokens:
            cands = self.lmor.get(w, [])
            cands_per_word.append(cands)

        return {
            "word_ids": word_ids,
            "lemma_ids": lemma_ids,
            "cands_per_word": cands_per_word,
            "seq_len": len(word_ids)
        }

def pad_collate_fn(batch: List[Dict], num_lemmas: int, pad_word_id: int = 0, pad_lemma_id: int = 0):
    """
    Collate function avec padding et construction du masque booléen 3D (B, T, |V_l|).
    Applique la règle anti-NaN : si un mot n'a aucun candidat, masque = tout True.
    """
    batch_size = len(batch)
    max_len = max(item["seq_len"] for item in batch)

    word_ids_tensor = torch.full((batch_size, max_len), pad_word_id, dtype=torch.long)
    lemma_ids_tensor = torch.full((batch_size, max_len), pad_lemma_id, dtype=torch.long)
    
    # Masque morphologique (B, T, |V_l|)
    candidate_mask = torch.zeros((batch_size, max_len, num_lemmas), dtype=torch.bool)

    fallback_oov_count = 0
    total_tokens_count = 0

    for i, item in enumerate(batch):
        slen = item["seq_len"]
        word_ids_tensor[i, :slen] = torch.tensor(item["word_ids"], dtype=torch.long)
        lemma_ids_tensor[i, :slen] = torch.tensor(item["lemma_ids"], dtype=torch.long)

        for t, cands in enumerate(item["cands_per_word"]):
            total_tokens_count += 1
            if not cands:
                # Sécurité anti-NaN : mot hors lexique ou sans candidat -> fallback neutre tout à True
                candidate_mask[i, t, :] = True
                fallback_oov_count += 1
            else:
                candidate_mask[i, t, cands] = True

        # Pour les positions de padding <PAD>, masque à True pour éviter toute division par zéro / NaN
        if slen < max_len:
            candidate_mask[i, slen:, :] = True

    return {
        "word_ids": word_ids_tensor,
        "lemma_ids": lemma_ids_tensor,
        "candidate_mask": candidate_mask,
        "lengths": [item["seq_len"] for item in batch],
        "fallback_count": fallback_oov_count,
        "total_tokens": total_tokens_count
    }
