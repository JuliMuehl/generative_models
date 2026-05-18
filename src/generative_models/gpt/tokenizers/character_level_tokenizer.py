import torch

class CharacterLevelTokenizer:
    def __init__(self, training_corpus):
        self.vocab = sorted(list(set(training_corpus))) + ["EOF"]
        self.char_to_int = {c : i for i,c in enumerate(self.vocab)}
        self.int_to_char = {i : c for i,c in enumerate(self.vocab)}

    def encode(self, s, eof=False):
        return torch.tensor(
                [self.char_to_int[c] for c in s] + ([self.char_to_int["EOF"]] if eof else []), 
                dtype=torch.long
        )

    def decode(self, token_batch):
        return [''.join(self.int_to_char[token_batch[i,j].item()]  for j in range(token_batch.shape[1])) for i in range(token_batch.shape[0])]

    @property
    def vocab_size(self):
        return len(self.vocab)

