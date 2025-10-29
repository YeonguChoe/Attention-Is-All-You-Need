from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import torch

def tokenize(s:str):
    word_level = models.WordLevel(unk_token="[UNK]")
    tokenizer = Tokenizer(model=word_level)
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train(files=["wiki.train.raw"], trainer=trainer)
    tokenizer.save("vocab.json")
    return tokenizer.encode(s).ids