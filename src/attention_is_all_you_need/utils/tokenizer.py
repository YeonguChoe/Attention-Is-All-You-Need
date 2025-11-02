import torch
from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Whitespace

class CustomTokenizer:
    def __init__(self, path="vocab.json"):
        self.path = path
        self.tokenizer = None
    def train(self, files=["wiki.train.raw"]):
        word_level = models.WordLevel(unk_token="[UNK]")
        self.tokenizer = Tokenizer(model=word_level)
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train(files=files, trainer=trainer)
        self.tokenizer.save(self.path)  # save to vocab.json
    def load(self):
        self.tokenizer = Tokenizer.from_file(self.path)
    def tokenize(self,s: str) -> torch.Tensor:
        if self.tokenizer is None:
            self.load()
        encoded_s = self.tokenizer.encode(s)
        return torch.tensor(data=encoded_s.ids, dtype=torch.long)
    def get_vocab_size(self):
        if self.tokenizer is None:
            self.load()
        return self.tokenizer.get_vocab_size()
    

# Example
tokenizer = CustomTokenizer()
tokenizer.train()

tokens = tokenizer.tokenize("Hello from Canada")
print(tokens)

vocab_size = tokenizer.get_vocab_size()
print(f"vocabulary size: {vocab_size}")