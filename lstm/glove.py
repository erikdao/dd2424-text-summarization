import typing
import torch
import torchtext.vocab as vocab
from torchtext.data import get_tokenizer

from config import config

glove = vocab.GloVe(name='6B', dim=50)

def tokenize(sentence: typing.Optional[str], tokenizer: typing.Optional[typing.Any]) -> typing.List[typing.Any]:
    sentence = sentence.strip().lower().replace("\n", "")
    punctuation = ['(', ')', ':', '"', ' ']
    return [tok for tok in tokenizer(sentence) if (not tok.isspace() and tok not in punctuation)]

# def main():
#     glove = vocab.GloVe(name='6B', dim=50)

#     print('Loaded {} words'.format(len(glove.itos)))

#     tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
#     input_sentence = "these macaroons are better than the non vegan raw ones! i swear it! the blondes too   they taste like they are made with butter. truly a great product! buy people and you will be hooked  in a good way ,too good for words!"
#     tokens = tokenize(input_sentence, tokenizer=tokenizer)
#     print(tokens)

#     print(config.HIDDEN_SIZE)

def main():
    print(glove.itos[0], glove.itos[1])
if __name__ == '__main__':
    main()
