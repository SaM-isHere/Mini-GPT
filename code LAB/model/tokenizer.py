import json
from collections import Counter

class SimpleTokenizer:
    def __init__(self, min_freq=2):
        self.stoi = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3}
        self.itos = {v:k for k,v in self.stoi.items()}
        self.min_freq = min_freq

    def train(self, text):
        counter = Counter(text.split())
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

    def encode(self, text):
        return [self.stoi["<bos>"]] + [
            self.stoi.get(w, self.stoi["<unk>"]) for w in text.split()
        ] + [self.stoi["<eos>"]]

    def decode(self, tokens):
        return " ".join(self.itos.get(t, "<unk>") for t in tokens)\
            .replace("<bos>", "").replace("<eos>", "").strip()

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    def load(self, path):
        with open(path) as f:
            self.stoi = json.load(f)
        self.itos = {int(v):k for k,v in self.stoi.items()}
