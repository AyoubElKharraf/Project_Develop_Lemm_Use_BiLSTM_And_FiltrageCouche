import torch
import pytest
from src.models import BiLSTMBaseline, BiLSTMFiltered

def test_baseline_shapes():
    vocab_w = 100
    vocab_l = 50
    model = BiLSTMBaseline(vocab_w, vocab_l, embed_dim=32, hidden_dim=64)
    x = torch.randint(0, vocab_w, (4, 10)) # B=4, T=10
    logits = model(x)
    assert logits.shape == (4, 10, vocab_l)

def test_filtered_logits_masking():
    vocab_w = 100
    vocab_l = 10
    model = BiLSTMFiltered(vocab_w, vocab_l, embed_dim=32, hidden_dim=64)
    x = torch.randint(0, vocab_w, (2, 5)) # B=2, T=5
    
    # Création d'un masque où seuls les indices 2 et 4 sont autorisés
    mask = torch.zeros((2, 5, vocab_l), dtype=torch.bool)
    mask[:, :, 2] = True
    mask[:, :, 4] = True

    logits = model(x, mask=mask)
    probs = torch.softmax(logits, dim=-1)

    # Vérification que les probabilités des lemmes non candidats sont strictement nulles
    for idx in [0, 1, 3, 5, 6, 7, 8, 9]:
        assert torch.all(probs[:, :, idx] < 1e-6)

    # Vérification de la somme probabiliste égale à 1 sur les candidats
    assert torch.allclose(probs.sum(dim=-1), torch.ones((2, 5)), atol=1e-5)

def test_anti_nan_safety():
    vocab_w = 50
    vocab_l = 10
    model = BiLSTMFiltered(vocab_w, vocab_l, embed_dim=16, hidden_dim=32)
    x = torch.randint(0, vocab_w, (1, 3))
    
    # Règle anti-NaN : si aucun candidat (fallback neutre tout True)
    mask = torch.ones((1, 3, vocab_l), dtype=torch.bool)
    logits = model(x, mask=mask)
    loss = logits.sum()
    loss.backward()

    # Les gradients ne doivent comporter aucun NaN
    for p in model.parameters():
        assert not torch.isnan(p.grad).any()
