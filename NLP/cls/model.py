import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.2):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, lstm_layers=1, dropout=0.2):
        super(ConvClassifier, self).__init__()
        # self.embedding = nn.Embedding(seq_len, embedding_dim)
        self.conv1 = ConvLayer(embedding_dim, hidden_dim, kernel_size=3, padding=1, dropout=dropout)
        self.conv2 = ConvLayer(embedding_dim, hidden_dim, kernel_size=5, padding=2, dropout=dropout)

        self.use_lstm = lstm_layers > 0
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim * 2,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0
            )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)        # (batch_size, embedding_dim, seq_len)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), 1)  # (batch_size, hidden_dim * 2, seq_len)
        if self.use_lstm:
            x = x.permute(0, 2, 1)        # (batch_size, seq_len, hidden_dim * 2)
            x, _ = self.lstm(x)           # (batch_size, seq_len, hidden_dim * 2)
            x = x.permute(0, 2, 1)        # (batch_size, hidden_dim * 2, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        x = self.fc(x)                # (batch_size, num_classes)
        return x