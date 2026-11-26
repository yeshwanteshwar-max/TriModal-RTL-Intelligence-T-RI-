import torch


class VCDEncoder(torch.nn.Module):
    """Encode fixed-width VCD feature sequences into waveform embeddings."""

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.15,
        normalize_embedding: bool = True,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0 or num_layers <= 0:
            raise ValueError("input_dim, hidden_dim, output_dim, and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.normalize_embedding = normalize_embedding
        self.bidirectional = bidirectional
        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )
        self.gru = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        recurrent_dim = hidden_dim * (2 if bidirectional else 1)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(recurrent_dim * 2, output_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("VCD features must have shape [batch, time, feature_dim]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"VCD feature dimension must be {self.input_dim}")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("VCD feature batches and sequences must be non-empty")

        batch_size, sequence_length, _ = features.shape
        if lengths is not None:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device="cpu")
            if lengths.ndim != 1 or lengths.shape[0] != batch_size:
                raise ValueError("lengths must have shape [batch_size]")
            if torch.any(lengths <= 0) or torch.any(lengths > sequence_length):
                raise ValueError("lengths must be between 1 and the padded sequence length")

        projected = self.input_projection(features)
        if lengths is None:
            output, _ = self.gru(projected)
            valid_lengths = torch.full(
                (batch_size,), sequence_length, dtype=torch.long, device=features.device
            )
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                projected,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.gru(packed)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=sequence_length,
            )
            valid_lengths = lengths.to(features.device)

        mask = (
            torch.arange(sequence_length, device=features.device)[None, :]
            < valid_lengths[:, None]
        )
        masked_output = output.masked_fill(~mask[:, :, None], 0.0)
        mean_pool = masked_output.sum(dim=1) / valid_lengths[:, None].to(output.dtype)
        max_pool = output.masked_fill(~mask[:, :, None], torch.finfo(output.dtype).min).amax(dim=1)
        embedding = self.projection(torch.cat((mean_pool, max_pool), dim=-1))
        if self.normalize_embedding:
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding
