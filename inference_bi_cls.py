# -*- coding: utf - 8 -*-

import json
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from utils.distribu_utils import get_distribution_strategy
from utils.distribu_utils import get_strategy_scope
from create_models import create_binary_cls_model
from data_processors.commom import extract_examples_dict_from_relation_questions

distribution_strategy = get_distribution_strategy('one_device')

with get_strategy_scope(distribution_strategy):

    model = create_binary_cls_model(is_train=False, use_pretrain=False)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(
        checkpoint_dir='saved_models/binary_cls_models'
    ))

# 对 <阅读理解> 第二步推断出的结果进行预测（过滤掉非法结果）
valid_data_path = 'inference_results/mrc_results/in_use/second_step/valid_results.json'

with tf.io.gfile.GFile(valid_data_path, mode='r') as reader:
    valid_data = json.load(reader)
reader.close()

MAX_SEQ_LEN = 165
paragraphs = valid_data['data'][0]['paragraphs']
relation_questions_dict = extract_examples_dict_from_relation_questions()
tokenizer = BertWordPieceTokenizer(vocab_file='vocabs/bert-base-chinese-vocab.txt')
tokenizer.enable_padding(length=MAX_SEQ_LEN)
tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)

print(len(paragraphs))

for index, paragraph in enumerate(paragraphs):
    context = paragraph['context']
    pred_sros = paragraph['pred_sros']

    filtered_relations = []
    for sro in pred_sros:
        _relation = sro['relation']
        _subject = sro['subject']
        _object = sro['object']

        relation_questions = relation_questions_dict[_relation]

        question_c = relation_questions.question_c

        question_c = question_c.replace('subject', _subject).replace('object', _object)

        tokenizer_output = tokenizer.encode(question_c, context)

        inputs_ids = tokenizer_output.ids
        inputs_mask = tokenizer_output.attention_mask
        segment_ids = tokenizer_output.type_ids

        inputs_ids = tf.reshape(tf.constant(inputs_ids), (1, -1))
        inputs_mask = tf.reshape(tf.constant(inputs_mask), (1, -1))
        segment_ids = tf.reshape(tf.constant(segment_ids), (1, -1))

        # 推断一个结果
        model_output = model.predict({
            'inputs_ids': inputs_ids,
            'inputs_mask': inputs_mask,
            'segment_ids': segment_ids
        })

        prob = model_output[0][0]

        if prob >= 0.3:
            filtered_relations.append({
                'relation': _relation,
                'subject': _subject,
                'object': _object
            })

    paragraph['pred_sros'] = filtered_relations
    print(index)


results = {
    'data': [{'paragraphs': paragraphs}]
}
output_save_path = 'inference_results/bi_cls_results/valid_results.json'
with tf.io.gfile.GFile(output_save_path, mode='w') as writer:
    writer.write(json.dumps(results, ensure_ascii=False, indent=2))
writer.close()

