import torch
import torch.nn as nn

class BiLSTMBaseline(nn.Module):
    """
    BiLSTM standard pour la lemmatisation arabe (sans filtrage).
    Architecture de référence du cours de Master M2I :
    - Embedding Layer (vocab_size -> embed_dim 128)
    - BiLSTM Layer (hidden_dim 128 par direction -> 256 en sortie)
    - Dropout
    - Linear Projection (256 -> |V_l|)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) - identifiants de mots
        Retourne : logits bruts (B, T, |V_l|)
        """
        emb = self.embedding(x)             # (B, T, embed_dim)
        lstm_out, _ = self.bilstm(emb)       # (B, T, hidden_dim * 2)
        out = self.dropout(lstm_out)
        logits = self.fc(out)                # (B, T, |V_l|)
        return logits
