# Attention Is All You Need

## Encoder
- Role: Convert sentence into word vector.
- Word vector contains meaning of word

## Decoder
- Role: Generate next word using previous word and word vector from encoder.

## Input Embedding
- Add meaning of the word.
- It is the meaning of a word itself. Not word meaning in the sentence.

## Positional Encoding
- Add position of word information within a sentence.
- It doesn't represent the relation between words in a sentence.
- It only encode position of word in a sentence.

## Input Embedding + Positional Encoding
- Reason for addition: Encode word meaning and position within sentence, in one vector (torch.tensor).

## Batch
- sentences within an input tensor.
## Sequence
- word (token) within a sentence.
## Token
- Word
## Vector
- list like data structure that represent property of object
- Tensor is to represent vector.
- Property example: square foot, bed room, bath room
- Example: [32, 21, 55]

