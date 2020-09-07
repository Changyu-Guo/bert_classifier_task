# -*- coding: utf - 8 -*-

from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer('vocab.txt')

out = tokenizer.encode('我爱“中国”')

print(out.tokens)