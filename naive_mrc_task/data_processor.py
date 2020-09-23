# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base

    Convert MRC data to squad format.
"""

import copy
import json
import collections
import tensorflow as tf
from absl import logging

import tokenization
from common_data_utils import get_squad_json_qas_item_template
from common_data_utils import extract_examples_dict_from_relation_questions


def convert_last_step_results_for_train(results_path, save_path):
    """
        将上一步的推断结果转换为本步骤训练需要的数据
        在训练步骤中，所有问题都存在答案

        对于每个 context, 对应多个 relation
        每个 relation 都有两个问答
        所以当前 context 的 qas 中会有 2 * relation_num 个条目
    """
    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )

    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    paragraphs = results['data'][0]['paragraphs']

    _id = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        context = paragraph['context']
        pred_sros = paragraph['pred_sros']
        for sro in pred_sros:
            s = sro['subject']
            o = sro['object']

            s_start_pos = context.find(s)
            o_start_pos = context.find(o)

            relation_questions = relation_questions_dict[sro['relation']]
            question_a = relation_questions.question_a
            question_b = relation_questions.question_b.replace('subject', s)

            squad_json_qas_item = get_squad_json_qas_item_template(
                question=question_a,
                answers=[{
                    'text': s,
                    'answer_start': s_start_pos
                }],
                qas_id='id_' + str(_id)
            )
            _id += 1
            paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

            squad_json_qas_item = get_squad_json_qas_item_template(
                question=question_b,
                answers=[{
                    'text': o,
                    'answer_start': o_start_pos
                }],
                qas_id='id_' + str(_id)
            )
            _id += 1
            paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

    results['data'][0]['paragraphs'] = paragraphs
    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(results, ensure_ascii=False, indent=2))
    writer.close()


def convert_last_step_results_for_infer(results_path, save_path, step='first'):

    if step not in ['first', 'second']:
        raise ValueError('step must be first or second')

    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )

    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    paragraphs = results['data'][0]['paragraphs']

    qas_id = 0
    for paragraph_index, paragraph in enumerate(paragraphs):

        # 在 first step 中使用 relation 推断 subject
        # 在 second step 中使用 relation 和 subject 推断 object
        pred_sros = paragraph['pred_sros']

        for sro_index, sro in enumerate(pred_sros):
            # relation 是一定存在的一个键
            relation_questions = relation_questions_dict[sro['relation']]
            if step == 'first':
                question = relation_questions.question_a
            if step == 'second':
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
            question_text,
            doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=False
    ):
        self.qas_id = qas_id

        # 在推断的过程中, 根据 paragraph_index 确定当前 example 所属的 context
        # 接下来再根据 qas_id 确定当前 example 对应的 sro
        # 从而将答案填入相应的位置
        self.paragraph_index = paragraph_index
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
        self._writer = tf.io.TFRecordWriter(filename)

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


def read_squad_examples(input_file, is_training):

    with tf.io.gfile.GFile(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
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

            # 这里需要注意
            # train data 有 answer
            if is_training:

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
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
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
        batch_size=None
):

    base_id = 1000000000
    unique_id = base_id
    feature = None
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
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
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"]
        )
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
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
                is_impossible=example.is_impossible)

            # Run callback
            if is_training:
                output_fn(feature)
            else:
                output_fn(feature, is_padding=False)

            unique_id += 1

    # if not is_training and feature:
    #     assert batch_size
    #     num_padding = 0
    #     num_examples = unique_id - base_id
    #     if unique_id % batch_size != 0:
    #         num_padding = batch_size - (num_examples % batch_size)
    #     logging.info("Adding padding examples to make sure no partial batch.")
    #     logging.info("Adds %d padding examples for inference.", num_padding)
    #     dummy_feature = copy.deepcopy(feature)
    #     for _ in range(num_padding):
    #         dummy_feature.unique_id = unique_id
    #
    #         # Run callback
    #         output_fn(feature, is_padding=True)
    #         unique_id += 1

    return unique_id - base_id


def generate_tfrecord_from_json_file(
        input_file_path,
        vocab_file_path,
        output_file_path,
        max_seq_len=384,
        max_query_len=64,
        doc_stride=128,
        is_train=True
):
    # 从原始数据中读取所有的 example
    examples = read_squad_examples(
        input_file=input_file_path,
        is_training=is_train
    )

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path)
    writer = FeatureWriter(filename=output_file_path, is_training=is_train)

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
        output_fn=_append_feature,
        batch_size=None
    )

    meta_data = {
        'data_size': writer.num_features,
        'max_seq_len': max_seq_len,
        'max_query_len': max_query_len,
        'doc_stride': doc_stride
    }

    return meta_data


if __name__ == '__main__':
    read_squad_examples(
        input_file='datasets/raw/for_infer/temp_valid.json',
        is_training=False
    )
