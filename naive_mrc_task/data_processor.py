# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base

    Convert MRC data to squad format.
"""

import os
import six
import math
import gzip
import json
import pickle
import collections
import tensorflow as tf
from absl import logging

import tokenization
from common_data_utils import get_squad_json_template
from common_data_utils import extract_examples_dict_from_relation_questions


def convert_origin_data_for_train(origin_data_path, save_path):
    """
        将原始的数据转换为当前步骤的训练数据, 即只使用正确的数据来训练模型, 不考虑其他的情况
    """
    # relation -> questions
    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )

    with tf.io.gfile.GFile(origin_data_path, mode='r') as reader:
        init_train_examples = json.load(reader)
    reader.close()

    squad_json = get_squad_json_template(title='naive mrc task')

    qas_id = 0
    for init_train_example in init_train_examples:
        text = init_train_example['text']
        sros = init_train_example['sros']

        squad_json_paragraph = {
            'context': text,
            'qas': []
        }

        for sro_index, sro in enumerate(sros):
            s = sro['subject']
            r = sro['relation']
            o = sro['object']

            # 确定第一个问题的答案开始位置
            s_start_pos = text.find(s)

            # 确定第二个问题的答案开始位置
            o_start_pos = text.find(o)

            relation_questions = relation_questions_dict[r]

            question_a = relation_questions.question_a
            question_b = relation_questions.question_b.replace('subject', s)

            squad_json_qas_item = {
                'question': question_a,
                'answers': [{
                    'text': s,
                    'answer_start': s_start_pos
                }],
                'id': 'id_' + str(qas_id)
            }
            qas_id += 1
            squad_json_paragraph['qas'].append(squad_json_qas_item)

            squad_json_qas_item = {
                'question': question_b,
                'answers': [{
                    'text': o,
                    'answer_start': o_start_pos
                }],
                'id': 'id_' + str(qas_id)
            }
            qas_id += 1
            squad_json_paragraph['qas'].append(squad_json_qas_item)

        squad_json['data'][0]['paragraphs'].append(squad_json_paragraph)

    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(squad_json, ensure_ascii=False, indent=2) + '\n')
    writer.close()


def convert_origin_data_for_infer(origin_data_path, save_path, step='1'):
    """
        将原始数据转换为当前步骤的推断数据, 用于查看训练好的模型在正确数据上的 F 值 和 EM
    """

    if step not in ['1', '2', '12']:
        raise ValueError('step must be 1 or 2 or 12')

    # relation -> questions
    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )

    with tf.io.gfile.GFile(origin_data_path, mode='r') as reader:
        init_train_examples = json.load(reader)
    reader.close()

    squad_json = get_squad_json_template(title='naive mrc task')

    qas_id = 0
    for init_train_example in init_train_examples:

        text = init_train_example['text']
        sros = init_train_example['sros']

        squad_json_paragraph = {
            'context': text,
            'qas': [],
            'origin_sros': sros,
            'pred_sros': []
        }

        for sro_index, sro in enumerate(sros):
            s = sro['subject']
            r = sro['relation']
            o = sro['object']

            squad_json_paragraph['pred_sros'].append({'relation': r})

            relation_questions = relation_questions_dict[r]

            # 确定第一个问题的答案开始位置
            s_start_pos = text.find(s)

            # 确定第二个问题的答案开始位置
            o_start_pos = text.find(o)

            question_a = relation_questions.question_a
            question_b = relation_questions.question_b.replace('subject', s)

            # 第一个推断步骤中只使用到第一个问题
            if step == '1':
                squad_json_qas_item = {
                    'question': question_a,
                    'answers': [{
                        'text': s,
                        'answer_start': s_start_pos
                    }],
                    'id': 'id_' + str(qas_id),
                    'sro_index': sro_index,
                }
                qas_id += 1
                squad_json_paragraph['qas'].append(squad_json_qas_item)

            if step == '2':
                squad_json_qas_item = {
                    'question': question_b,
                    'answers': [{
                        'text': o,
                        'answer_start': o_start_pos
                    }],
                    'id': 'id_' + str(qas_id),
                    'sro_index': sro_index
                }
                qas_id += 1
                squad_json_paragraph['qas'].append(squad_json_qas_item)

            if step == '12':
                squad_json_qas_item = {
                    'question': question_a,
                    'answers': [{
                        'text': s,
                        'answer_start': s_start_pos
                    }],
                    'question_type': '1',
                    'id': 'id_' + str(qas_id),
                    'sro_index': sro_index
                }
                qas_id += 1
                squad_json_paragraph['qas'].append(squad_json_qas_item)

                squad_json_qas_item = {
                    'question': question_b,
                    'answers': [{
                        'text': o,
                        'answer_start': o_start_pos
                    }],
                    'question_type': '2',
                    'id': 'id_' + str(qas_id),
                    'sro_index': sro_index
                }
                qas_id += 1
                squad_json_paragraph['qas'].append(squad_json_qas_item)

        squad_json['data'][0]['paragraphs'].append(squad_json_paragraph)

    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(squad_json, ensure_ascii=False, indent=2) + '\n')
    writer.close()


def convert_last_step_results_for_train(results_path, save_path, step='12'):
    """
        将上一步的推断结果转换为本步骤训练需要的数据
        由于使用的是上一步骤的推断结果，因此存在没有答案的情况

        当前不考虑 由 relation 和 subject 预测 object 构造不出问题的情况
        这类问题直接不构造, 只考虑 由 relation 预测 subject 没有答案的情况
    """

    if step not in ['1', '2', '12']:
        raise ValueError('step must be 1 or 2 or 12')

    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )

    # 加载上一步骤的推断结果
    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    paragraphs = results['data'][0]['paragraphs']

    qas_id = 0
    for paragraph_index, paragraph in enumerate(paragraphs):

        paragraphs[paragraph_index]['qas'] = []

        context = paragraph['context']

        origin_sros = paragraph['origin_sros']
        origin_relations = set([sro['relation'] for sro in origin_sros])

        pred_sros = paragraph['pred_sros']

        # origin sro 存在完整的 s / r / o
        # 因此可以构建出两个问答
        for index, sro in enumerate(origin_sros):
            s = sro['subject']
            r = sro['relation']
            o = sro['object']

            # 确定第一个问题的答案开始位置
            s_start_pos = context.find(s)

            # 确定第二个问题的答案开始位置
            o_start_pos = context.find(o)

            relation_questions = relation_questions_dict[r]

            # 第一个问题不需要进行关键词替换
            question_a = relation_questions.question_a

            # 第二个问题需要将 subject 替换为当前的真实 subject
            question_b = relation_questions.question_b.replace('subject', s)

            if step == '1':
                # 构建第一个问题的 qas item
                squad_json_qas_item = {
                    'question': question_a,
                    'answers': [{
                        'text': s,
                        'answer_start': s_start_pos
                    }],
                    'is_impossible': False,
                    'id': 'id_' + str(qas_id)
                }
                qas_id += 1
                paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

            if step == '2':
                # 构建第二个问题的 qas item
                squad_json_qas_item = {
                    'question': question_b,
                    'answers': [{
                        'text': o,
                        'answer_start': o_start_pos
                    }],
                    'is_impossible': False,
                    'id': 'id_' + str(qas_id)
                }
                qas_id += 1
                paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

            if step == '12':
                # 构建第一个问题的 qas item
                squad_json_qas_item = {
                    'question': question_a,
                    'answers': [{
                        'text': s,
                        'answer_start': s_start_pos
                    }],
                    'is_impossible': False,
                    'id': 'id_' + str(qas_id)
                }
                qas_id += 1
                paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

                # 构建第二个问题的 qas item
                squad_json_qas_item = {
                    'question': question_b,
                    'answers': [{
                        'text': o,
                        'answer_start': o_start_pos
                    }],
                    'is_impossible': False,
                    'id': 'id_' + str(qas_id)
                }
                qas_id += 1
                paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

        # 负样本
        for index, sro in enumerate(pred_sros):
            r = sro['relation']

            # 跳过正样本
            if r in origin_relations:
                continue

            relation_questions = relation_questions_dict[r]

            if step == '1' or step == '12':
                question_a = relation_questions.question_a
                squad_json_qas_item = {
                    'question': question_a,
                    'answers': [],
                    'is_impossible': True,
                    'id': 'id_' + str(qas_id)
                }
                qas_id += 1
                paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

            if step == '2':
                # 上一步预测本应该为空, 但是被预测出来的结果
                question_b = relation_questions.question_b.replace('subject', sro['subject'])
                squad_json_qas_item = {
                    'question': question_b,
                    'answers': [],
                    'is_impossible': True,
                    'id': 'id_' + str(qas_id)
                }
                qas_id += 1
                paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

    results['data'][0]['paragraphs'] = paragraphs
    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    writer.close()


def convert_last_step_results_for_infer(results_path, save_path, step='1'):
    """
        上一步骤对数据进行了推断之后, 要经过当前步骤的模型进行推断
        上一步骤推断出了原始文本应该包含的所有 relation, 这些 relation 中包含一些错误的 relation
        当前步骤要做的就是根据所有推断出的 relation (正确的 + 错误的) 推断出 subject 和 object
        推断分两个步骤:
            1. 第一步是根据 relation 推断出 subject
            2. 第二步是根据 relation 和 subject 推断出 object
        注意: 在完成第一步推断之后需要完成不存在答案的结果的过滤

        因此在准备推断数据的时候, 也需要分两个阶段进行准备:
            1. 首先准备第一阶段的数据, 该阶段的数据问题只包含 question_a
            2. 然后准备第二阶段的数据, 该阶段的数据问题只包含 question_b

        准备好数据之后由模型对这些问题进行推断
    """

    if step not in ['1', '2']:
        raise ValueError('step must be 1 or 2')

    # relation -> questions
    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )

    # 获取上一步骤的推断结果
    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    paragraphs = results['data'][0]['paragraphs']

    # 开始构建问题
    qas_id = 0
    for paragraph_index, paragraph in enumerate(paragraphs):

        paragraphs[paragraph_index]['qas'] = []

        # 在 first step 中使用 relation 推断 subject
        # 在 second step 中使用 relation 和 subject 推断 object
        pred_sros = paragraph['pred_sros']

        for sro_index, sro in enumerate(pred_sros):

            relation_questions = relation_questions_dict[sro['relation']]

            # 第一步推断使用第一个问题
            if step == '1':
                question = relation_questions.question_a

            # 第二步推断使用第二个问题
            if step == '2':
                question = relation_questions.question_b.replace('subject', sro['subject'])

            squad_json_qas_item = {
                'question': question,
                'id': 'id_' + str(qas_id),
                'sro_index': sro_index
            }
            qas_id += 1
            paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

    results['data'][0]['paragraphs'] = paragraphs
    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(results, ensure_ascii=False, indent=2))
    writer.close()


class SquadExample:

    def __init__(
            self,
            qas_id,
            paragraph_index,
            sro_index,
            question_text,
            doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=False
    ):
        self.qas_id = qas_id

        # 在推断的过程中, 根据 paragraph_index 确定当前 example 所属的 context
        # 接下来再根据 sro_index 确定当前 example 对应的 sro
        # 从而将答案填入相应的位置
        self.paragraph_index = paragraph_index
        self.sro_index = sro_index
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class Feature:

    def __init__(
            self,

            # 用于和推断的结果相匹配
            unique_id,

            # 用于和 example 相匹配
            example_index,
            doc_span_index,
            tokens,
            token_to_orig_map,
            token_is_max_context,
            inputs_ids,
            inputs_mask,
            segment_ids,
            start_position=None,
            end_position=None,
            is_impossible=None
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.inputs_ids = inputs_ids
        self.inputs_mask = inputs_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class FeatureWriter:

    def __init__(self, filename, is_training):

        self.filename = filename

        options = tf.io.TFRecordOptions(compression_type='GZIP')
        self._writer = tf.io.TFRecordWriter(filename, options=options)

        self.is_training = is_training

        self.num_features = 0

    def process_feature(self, feature):
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["example_indices"] = create_int_feature([feature.example_index])
        features["inputs_ids"] = create_int_feature(feature.inputs_ids)
        features["inputs_mask"] = create_int_feature(feature.inputs_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def _improve_answer_span(
        doc_tokens,
        input_start,
        input_end,
        tokenizer,
        orig_answer_text
):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def read_squad_examples(input_file, is_training, version_2_with_negative):

    with tf.io.gfile.GFile(input_file, "r") as reader:
        input_data = json.load(reader)["data"]
    reader.close()

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 0xa0 or c == u'\u3000':
            return True
        return False

    examples = []
    paragraphs = input_data[0]['paragraphs']
    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        # 每个 QA 都构成一个 example
        for qa in paragraph["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]

            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False

            if is_training:
                sro_index = None
            else:
                sro_index = qa['sro_index']

            # 这里需要注意
            # train data 有 answer
            if is_training:

                if version_2_with_negative:
                    is_impossible = qa['is_impossible']

                if (len(qa["answers"]) != 1) and (not is_impossible):
                    raise ValueError(
                        "For training, each question should have exactly 1 answer.")

                if not is_impossible:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    actual_text = " ".join(doc_tokens[start_position: (end_position + 1)])
                    cleaned_answer_text = " ".join(
                        tokenization.whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logging.warning(
                            "Could not find answer: '%s' vs. '%s'",
                            actual_text, cleaned_answer_text
                        )
                        continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            example = SquadExample(
                qas_id=qas_id,
                paragraph_index=paragraph_index,
                sro_index=sro_index,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible
            )
            examples.append(example)

    return examples


def convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        output_fn,
):

    base_id = 1000000000
    unique_id = base_id
    for example_index, example in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0: max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for i, token in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text
            )

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"]
        )
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset

            # 长度过长, 需要分段
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc

            doc_spans.append(_DocSpan(start=start_offset, length=length))

            if start_offset + length == len(all_doc_tokens):
                break

            # 如果 doc_stride 小于 length, 则会产生重叠
            start_offset += min(length, doc_stride)

        for doc_span_index, doc_span in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index
                )
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            feature = Feature(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                inputs_ids=input_ids,
                inputs_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible
            )

            # Run callback
            if is_training:
                output_fn(feature)
            else:
                output_fn(feature)

            unique_id += 1

    return unique_id - base_id


def generate_tfrecord_from_json_file(
        input_file_path,
        vocab_file_path,
        tfrecord_save_path,
        meta_save_path,
        features_save_path,
        max_seq_len=384,
        max_query_len=64,
        doc_stride=128,
        is_train=True,
        version_2_with_negative=False
):
    # 从原始数据中读取所有的 example
    examples = read_squad_examples(
        input_file=input_file_path,
        is_training=is_train,
        version_2_with_negative=version_2_with_negative
    )

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path)
    writer = FeatureWriter(filename=tfrecord_save_path, is_training=is_train)

    features = []

    def _append_feature(feature):
        features.append(feature)
        writer.process_feature(feature)

    convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        doc_stride=doc_stride,
        max_query_length=max_query_len,
        is_training=is_train,
        output_fn=_append_feature
    )

    meta_data = {
        'data_size': writer.num_features,
        'max_seq_len': max_seq_len,
        'max_query_len': max_query_len,
        'doc_stride': doc_stride,
        'is_train': is_train,
        'version_2_with_negative': version_2_with_negative,
    }
    writer.close()

    # save meta
    with tf.io.gfile.GFile(meta_save_path, mode='w') as writer:
        writer.write(json.dumps(meta_data, ensure_ascii=False, indent=2) + '\n')
    writer.close()

    # save features
    with gzip.open(features_save_path, mode='wb') as writer:
        pickle.dump(features, writer, protocol=pickle.HIGHEST_PROTOCOL)
    writer.close()

    return meta_data


def get_final_text(pred_text, orig_text, do_lower_case, verbose=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose:
            logging.info("Unable to find text: '%s' in '%s'", pred_text, orig_text)
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose:
            logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                         orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose:
            logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose:
            logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def postprocess_results(
        raw_data_path,
        features_path,
        results_path,
        save_dir,
        prefix,
        n_best_size,
        max_answer_length,
        do_lower_case,
        step,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        verbose=False
):

    # 读取 raw data, 用于构建最终的结果
    with tf.io.gfile.GFile(raw_data_path, mode='r') as reader:
        raw_data = json.load(reader)
    reader.close()
    paragraphs = raw_data['data'][0]['paragraphs']

    examples = read_squad_examples(
        input_file=raw_data_path,
        is_training=False,
        version_2_with_negative=version_2_with_negative
    )

    with gzip.open(features_path, mode='rb') as reader:
        features = pickle.load(reader)
    reader.close()

    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    assert len(features) == len(results)

    # example_index 和 features 一对多
    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.example_index].append(feature)

    # unique_id 和 result 一对一
    unique_id_to_result = {}
    for result in results:
        unique_id_to_result[result['unique_id']] = result

    _PrelimPrediction = collections.namedtuple(
        'PrelimPrediction',
        ['feature_index', 'start_index', 'end_index', 'start_logit', 'end_logit']
    )

    all_nbest_json = collections.OrderedDict()

    # 用于 evaluate
    all_predictions = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    # 一个 example 就是一个 **问题 + 文本** 对应一条答案
    # 但是由于文本长度可能会很长需要进行切片处理
    # 所以一个样本会被处理成多个 feature
    # 因此在推断的过程中需要分别处理所有 feature, 并从中找出最合适的答案
    for example_index, example in enumerate(examples):

        # 获取当前 example 对应的所有 features
        cur_example_features = example_index_to_features[example_index]

        # 候选答案
        prelim_predictions = []

        score_null = 1000000
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0

        for feature_index, feature in enumerate(cur_example_features):

            # 获取当前 feature 对应的答案
            result = unique_id_to_result[feature.unique_id]

            # 获取最大的 n 个 logit 的位置
            n_best_start_index = _get_best_indexes(result['start_logits'], n_best_size)
            n_best_end_index = _get_best_indexes(result['end_logits'], n_best_size)

            if version_2_with_negative:
                feature_null_score = result['start_logits'][0] + result['end_logits'][0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result['start_logits'][0]
                    null_end_logit = result['end_logits'][0]

            # 对所有的 start_index 和 end_index 进行判断
            # 筛选出所有可能形成答案的 start_index 和 end_index
            for start_index in n_best_start_index:
                for end_index in n_best_end_index:

                    # 开始位置超过了 context 的长度, 非法 index
                    if start_index >= len(feature.tokens):
                        continue

                    # 结束位置超过了 context 的长度, 非法 index
                    if end_index >= len(feature.tokens):
                        continue

                    # 开始位置对应不到原文的位置, 非法 index
                    if start_index not in feature.token_to_orig_map:
                        continue

                    # 结束位置对应不到原文的位置, 非法 index
                    if end_index not in feature.token_to_orig_map:
                        continue

                    # 开始位置的 token 不是 max_content, 说明当前 feature 对应的 context 不在中间位置
                    if not feature.token_is_max_context.get(start_index, False):
                        continue

                    # 开始位置在结束位置后面, 非法 index
                    if end_index < start_index:
                        continue

                    # 通过上面的一系列过滤之后, 留下来的 index 都是合法的 index

                    # 计算答案长度
                    length = end_index - start_index + 1

                    # 超过了最长长度, 则舍弃这条答案
                    if length > max_answer_length:
                        continue

                    # 一个可能的答案的 index
                    # 将其保存下来
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result['start_logits'][start_index],
                            end_logit=result['end_logits'][end_index]
                        )
                    )

        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit
                )
            )

        # start_logit end_logit 的和越大就认为这个答案越好
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True
        )

        # 用于表示最好的 N 个预测结果之一
        _NbestPrediction = collections.namedtuple(
            'NbestPrediction', ['text', 'start_logit', 'end_logit']
        )

        # 已经见到过的预测结果
        seen_predictions = {}

        # 所有 n 个最好的结果
        nbest = []

        # 这里 for 循环的含义是:
        # 根据前面挑选出的最好的 n 个 (start_index, end_index) 找出合适的 **原文中的连续文本**
        for pred in prelim_predictions:
            # 如果已经找出 n 个最好的结果, 则停止后续的寻找
            if len(nbest) >= n_best_size:
                break

            # 当前预测结果来自的 feature
            feature = cur_example_features[pred.feature_index]

            # 有答案
            if pred.start_index > 0:

                # 答案对应的所有 tokens
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]

                # 答案在原始文章的 tokens 中的开始位置
                orig_doc_start = feature.token_to_orig_map[pred.start_index]

                # 答案在原始文章的 tokens 中的结束位置
                orig_doc_end = feature.token_to_orig_map[pred.end_index]

                # 答案在原始文章中对应的 tokens
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                # 将 tokens 合并
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                # 删除所有由 WordPiece 分词生成的 ##
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, do_lower_case=do_lower_case, verbose=verbose
                )

                # 如果当前答案已经出现过
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True

            # 没答案(预测为 -1)
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit
                )
            )

        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit
                    )
                )

        if not nbest:
            nbest.append(
                _NbestPrediction(text='empty', start_logit=0.0, end_logit=0.0)
            )

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['probability'] = probs[i]
            output['start_logit'] = entry.start_logit
            output['end_logit'] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
            final_answer = nbest_json[0]['text']
        else:
            if best_non_null_entry is not None:
                score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                    final_answer = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
                    final_answer = best_non_null_entry.text
            else:
                logging.warning("best_non_null_entry is None")
                scores_diff_json[example.qas_id] = score_null
                all_predictions[example.qas_id] = ""
                final_answer = ""

        all_nbest_json[example.qas_id] = nbest_json

        # 将答案写入 raw data -> paragraphs 中
        paragraph_index = example.paragraph_index
        sro_index = example.sro_index

        if step == 'first':
            paragraphs[paragraph_index]['pred_sros'][sro_index]['subject'] = final_answer
        elif step == 'second':
            paragraphs[paragraph_index]['pred_sros'][sro_index]['object'] = final_answer
        elif step == 'first_and_second':
            qas = paragraphs[paragraph_index]['qas']
            qas_id = example.qas_id
            for qa in qas:
                if qas_id == qa['id']:
                    question_type = qa['question_type']
                    if question_type == '1':
                        paragraphs[paragraph_index]['pred_sros'][sro_index]['subject'] = final_answer
                    elif question_type == '2':
                        paragraphs[paragraph_index]['pred_sros'][sro_index]['object'] = final_answer
                    else:
                        raise ValueError('Unknown question type')

                    break
        else:
            raise ValueError('step must be first or second')

    raw_data['data'][0]['paragraphs'] = paragraphs
    with tf.io.gfile.GFile(os.path.join(save_dir, prefix + 'results.json'), mode='w') as writer:
        writer.write(json.dumps(raw_data, ensure_ascii=False, indent=2) + '\n')
    writer.close()

    with tf.io.gfile.GFile(os.path.join(save_dir, prefix + 'all_predictions.json'), mode='w') as writer:
        writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=2) + '\n')
    writer.close()

    # with tf.io.gfile.GFile(os.path.join(save_dir, prefix + 'all_nbest.json'), mode='w') as writer:
    #     writer.write(json.dumps(all_nbest_json, ensure_ascii=False, indent=2) + '\n')
    # writer.close()

    # with tf.io.gfile.GFile(os.path.join(save_dir, prefix + 'scores_diff.json'), mode='w') as writer:
    #     writer.write(json.dumps(scores_diff_json, ensure_ascii=False, indent=2) + '\n')
    # writer.close()


def filter_results(results_path, save_path, step='1'):
    if step not in ['1', '2']:
        raise ValueError('step must be 1 or 2')

    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    paragraphs = results['data'][0]['paragraphs']

    for paragraph_index, paragraph in enumerate(paragraphs):

        paragraphs[paragraph_index]['qas'] = []

        pred_sros = paragraph['pred_sros']
        new_pred_sros = []

        for sro_index, sro in enumerate(pred_sros):

            if step == '1':
                if sro['subject'] != '':
                    new_pred_sros.append(sro)

            if step == '2':
                if sro['object'] != '':
                    new_pred_sros.append(sro)

        paragraphs[paragraph_index]['pred_sros'] = new_pred_sros

    results['data'][0]['paragraphs'] = paragraphs
    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    writer.close()


if __name__ == '__main__':

    # convert_origin_data_for_train(
    #     origin_data_path='../common-datasets/init-train-valid.json',
    #     save_path='datasets/version_3/train/valid.json',
    # )

    # convert_origin_data_for_infer(
    #     origin_data_path='../common-datasets/init-train-train.json',
    #     save_path='datasets/version_3/inference/origin/first_and_second/train.json',
    #     step='12'
    # )

    # convert_last_step_results_for_train(
    #     results_path='../multi_turn_mrc_cls_task/inference_results/version_1/postprocessed/valid_results.json',
    #     save_path='datasets/version_5/train/first/valid.json',
    #     step='1'
    # )

    # convert_last_step_results_for_infer(
    #     results_path='inference_results/version_4/last_version_1/first/postprocessed/filtered_valid_results.json',
    #     save_path='datasets/version_4/inference/last_version_1/second/valid.json',
    #     step='2'
    # )

    generate_tfrecord_from_json_file(
        input_file_path='datasets/version_5/train/first/train.json',
        vocab_file_path='../bert-base-chinese/vocab.txt',
        tfrecord_save_path='datasets/version_5/train/first/tfrecords/train.tfrecord',
        meta_save_path='datasets/version_5/train/first/meta/train_meta.json',
        features_save_path='datasets/version_5/train/first/features/train_features.pkl',
        max_seq_len=200,
        max_query_len=50,
        doc_stride=128,
        is_train=True,
        version_2_with_negative=True
    )

    # 推断第一步
    # postprocess_results(
    #     raw_data_path='datasets/version_4/inference/last_version_1/second/valid.json',
    #     features_path='datasets/version_4/inference/last_version_1/second/features/valid_features.pkl',
    #     results_path='inference_results/version_4/last_version_1/second/raw/valid_results.json',
    #     save_dir='inference_results/version_4/last_version_1/second/postprocessed',
    #     prefix='valid_',
    #     n_best_size=20,
    #     max_answer_length=10,
    #     do_lower_case=True,
    #     step='second',
    #     version_2_with_negative=True
    # )

    # filter_results(
    #     'inference_results/version_4/last_version_1/second/postprocessed/valid_results.json',
    #     'inference_results/version_4/last_version_1/second/postprocessed/filtered_valid_results.json'
    # )

    pass
