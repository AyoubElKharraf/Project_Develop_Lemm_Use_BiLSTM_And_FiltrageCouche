import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMFiltered(nn.Module):
    """
    BiLSTM Filtré avec couche de filtrage morphologique par Logit Masking.
    Architecture du cours :
    - Même encodeur BiLSTM que la baseline.
    - Projection linéaire vers les logits bruts s_t.
    - Application du masque morphologique M_t :
        s'_{t,j} = s_{t,j} si M_{t,j} == True
        s'_{t,j} = -1e9 (équivalent -inf) si M_{t,j} == False
    - Softmax probabiliste renormalisé sur les seuls lemmes valides.
    """
    def __init__(self, vocab_words_size: int, vocab_lemmas_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_words_size, embed_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_lemmas_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T) - tenseur d'indices de mots
        mask: (B, T, |V_l|) - masque booléen où True = lemme candidat compatible
        Retourne : logits masqués (B, T, |V_l|)
        """
        emb = self.embedding(x)             # (B, T, embed_dim)
        lstm_out, _ = self.bilstm(emb)       # (B, T, hidden_dim * 2)
        out = self.dropout(lstm_out)
        logits = self.fc(out)                # (B, T, |V_l|)

        if mask is not None:
            # Remplacement des logits des classes interdites par -1e9 (ou -inf)
            # Logit masking stabilisé pour CrossEntropyLoss et Softmax
            logits = logits.masked_fill(~mask, -1e9)

        return logits
