import torch
from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Whitespace


def tokenize(s: str) -> torch.Tensor:
    word_level = models.WordLevel(unk_token="[UNK]")
    tokenizer = Tokenizer(model=word_level)
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files=["wiki.train.raw"], trainer=trainer)
    tokenizer.save("vocab.json")  # save to vocab.json
    encoded_s = tokenizer.encode(s)
    return torch.tensor(data=encoded_s.ids, dtype=torch.long)


def get_vocab_size():
    tokenizer = Tokenizer.from_file("vocab.json")
    return tokenizer.get_vocab_size()
