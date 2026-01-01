import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        with open(f"{path}/text.txt", encoding="utf-8") as f:
            text = f.read()

        tokenizer.train(text)
        tokens = tokenizer.encode(text)

        self.data = []
        for i in range(len(tokens) - seq_len):
            self.data.append((
                tokens[i:i+seq_len],
                tokens[i+1:i+seq_len+1]
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)
