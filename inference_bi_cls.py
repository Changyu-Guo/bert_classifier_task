# -*- coding: utf - 8 -*-

import json
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from create_models import create_binary_cls_model
from data_processors.mrc_data_processor import extract_examples_from_relation_questions

model = create_binary_cls_model(is_train=False, use_pretrain=False)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(
    checkpoint_dir='saved_models/binary_cls_models'
))

valid_data_path = 'inference_results/mrc_results/in_use/second_step/valid_results.json'

with tf.io.gfile.GFile(valid_data_path, mode='r') as reader:
    valid_data = json.load(reader)
reader.close()

MAX_SEQ_LEN = 165
paragraphs = valid_data['data'][0]['paragraphs']
relation_questions_dict = extract_examples_from_relation_questions()
tokenizer = BertWordPieceTokenizer(vocab_file='vocabs/bert-base-chinese-vocab.txt')
tokenizer.enable_padding(length=MAX_SEQ_LEN)
tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)

for paragraph in paragraphs:
    context = paragraph['context']
    pred_relations = paragraph['pred_relations']

    filtered_relations = []
    for relation in pred_relations:
        _relation = relation['relation']
        _subject = relation['subject']
        _object = relation['object']

        relation_questions = relation_questions_dict[_relation]

        relation_question_c = relation_questions.relation_question_c

        relation_question_c = relation_question_c.replace('subject', _subject).replace('object', _object)

        tokenizer_output = tokenizer.encode(relation_question_c, context)

        inputs_ids = tokenizer_output.ids
        inputs_mask = tokenizer_output.attention_mask
        segment_ids = tokenizer_output.type_ids

        inputs_ids = tf.reshape(tf.constant(inputs_ids), (1, -1))
        inputs_mask = tf.reshape(tf.constant(inputs_mask), (1, -1))
        segment_ids = tf.reshape(tf.constant(segment_ids), (1, -1))

        model_output = model.predict({
            'input_ids': inputs_ids,
            'input_mask': inputs_mask,
            'segment_ids': segment_ids
        })

        prob = model_output['prob'][0][0]

        if prob >= 0.5:
            filtered_relations.append({
                'relation': _relation,
                'subject': _subject,
                'object': _object
            })

    paragraph['pred_relations'] = filtered_relations


results = {
    'data': [{'paragraphs': paragraphs}]
}
output_save_path = 'inference_results/bi_cls_results/valid_results.json'
with tf.io.gfile.GFile(output_save_path, mode='w') as writer:
    writer.write(json.dumps(results, ensure_ascii=False, indent=2))
writer.close()
