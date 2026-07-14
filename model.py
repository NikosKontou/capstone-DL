import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config as cfg


class SpinMultiTaskNet(nn.Module):
    """
    Causal sequence model: an LSTM only ever sees positions <= t when
    producing the prediction at position t (this is guaranteed by
    construction -- an LSTM processes the sequence left-to-right and
    each output only depends on past+current inputs, no future leakage
    like a bidirectional RNN or a non-causal Transformer would have).

    Predicts per-step spins_left / next_bet at every timestep, not just
    the last one, so the model gets a training signal from every valid
    position in every session.
    """

    def __init__(self, n_features: int, hidden_size: int = None,
                 num_layers: int = None, dropout: float = None,
                 head_hidden: int = None):
        super().__init__()
        hidden_size = hidden_size or cfg.RNN_HIDDEN_SIZE
        num_layers = num_layers or cfg.RNN_NUM_LAYERS
        dropout = cfg.RNN_DROPOUT if dropout is None else dropout
        head_hidden = head_hidden or cfg.HEAD_HIDDEN

        self.input_proj = nn.Linear(n_features, hidden_size)
        self.input_act = nn.ReLU()

        # dropout between LSTM layers only takes effect when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,  # bidirectional would leak future timesteps -- must stay False
        )
        self.head_dropout = nn.Dropout(dropout)

        self.head_spins_left = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1), nn.Softplus(),
        )
        self.head_next_bet = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1), nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (B, T, F) padded input
        lengths: (B,) true sequence lengths, used to pack the sequence
                 so the LSTM doesn't process (and isn't influenced by)
                 padding steps.
        Returns per-step predictions of shape (B, T, 1) each, padded
        back out to T with zeros wherever the input was padding (those
        positions are excluded from the loss via padding_mask anyway).
        """
        z = self.input_act(self.input_proj(x))

        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])

        out = self.head_dropout(out)
        pred_spins_left = self.head_spins_left(out)
        pred_next_bet = self.head_next_bet(out)
        return pred_spins_left, pred_next_bet


def build_model(n_features: int) -> SpinMultiTaskNet:
    return SpinMultiTaskNet(n_features=n_features)
