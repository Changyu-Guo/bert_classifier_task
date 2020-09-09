# -*- coding: utf - 8 -*-

import json
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from data_processor import extract_examples_from_init_train
from mrc_data_processor import extract_examples_from_relation_questions

tokenizer = BertWordPieceTokenizer('vocabs/bert-base-chinese-vocab.txt')


def count_text_len():
    init_train_examples = extract_examples_from_init_train()
    text_lens = []
    for example in init_train_examples:
        text = example.text
        tokenizer_output = tokenizer.encode(text)
        text_lens.append(len(tokenizer_output))
    sorted_text_lens = sorted(text_lens)

    # length 取 120 可以覆盖到 95% 的句子
    print(sorted_text_lens[int(len(sorted_text_lens) * 0.95)])


def count_question_len():
    mrc_train_json = 'datasets/preprocessed_datasets/mrc_train.json'
    mrc_valid_json = 'datasets/preprocessed_datasets/mrc_valid.json'

    with tf.io.gfile.GFile(mrc_train_json, mode='r') as reader:
        train_data = json.load(reader)['data'][0]
    reader.close()
    with tf.io.gfile.GFile(mrc_valid_json, mode='r') as reader:
        valid_data = json.load(reader)['data'][0]
    reader.close()

    train_paragraphs = train_data['paragraphs']
    valid_paragraphs = valid_data['paragraphs']

    question_lens = []
    for item in train_paragraphs:
        qas = item['qas']
        for qa in qas:
            question = qa['question']
            tokenizer_output = tokenizer.encode(question)
            question_lens.append(len(tokenizer_output))

    for item in valid_paragraphs:
        qas = item['qas']
        for qa in qas:
            question = qa['question']
            tokenizer_output = tokenizer.encode(question)
            question_lens.append(len(tokenizer_output))

    sorted_question_lens = sorted(question_lens)
    print(sorted_question_lens[-20:])
    print(sorted_question_lens[int(len(sorted_question_lens) * 0.999)])


if __name__ == '__main__':
    count_question_len()
