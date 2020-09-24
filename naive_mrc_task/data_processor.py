# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base

    Convert MRC data to squad format.
"""

import math
import gzip
import json
import pickle
import collections
import tensorflow as tf
from absl import logging

import tokenization
from common_data_utils import get_squad_json_qas_item_template
from common_data_utils import extract_examples_dict_from_relation_questions
from common_data_utils import read_init_train_train_examples


def convert_origin_data_for_train(origin_data_path, save_path):
    pass


def convert_origin_data_for_infer(origin_data_path, save_path):
    pass


def convert_last_step_results_for_train(results_path, save_path, step='first'):
    """
        将上一步的推断结果转换为本步骤训练需要的数据
        由于使用的是上一步骤的推断结果，因此存在没有答案的情况

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

        origin_sros = paragraph['origin_sros']
        origin_relations = set([sro['relation'] for sro in origin_sros])

        pred_sros = paragraph['pred_sros']
        pred_relations = set([sro['relation'] for sro in pred_sros])

        print(origin_relations)
        print(pred_relations)

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

            # 第一步推断使用第一个问题
            if step == 'first':
                question = relation_questions.question_a

            # 第二步推断使用第二个问题
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


def read_squad_examples(input_file, is_training):

    with tf.io.gfile.GFile(input_file, "r") as reader:
        input_data = json.load(reader)["data"]
    reader.close()

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
            sro_index = qa['sro_index']

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
                output_fn(feature)

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
        tfrecord_save_path,
        meta_save_path,
        features_save_path,
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
        output_fn=_append_feature,
        batch_size=None
    )

    meta_data = {
        'data_size': writer.num_features,
        'max_seq_len': max_seq_len,
        'max_query_len': max_query_len,
        'doc_stride': doc_stride
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


def postprocess_results(
        raw_data_path,
        features_path,
        results_path,
        save_path,
        n_best_size=20,
        max_answer_length=30
):

    # 读取 raw data, 用于构建最终的结果
    with tf.io.gfile.GFile(raw_data_path, mode='r') as reader:
        raw_data = json.load(reader)
    reader.close()
    paragraphs = raw_data['data'][0]['paragraphs']

    examples = read_squad_examples(
        input_file=raw_data_path,
        is_training=False
    )

    with gzip.open(features_path, mode='rb') as reader:
        features = pickle.load(reader)
    reader.close()

    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    assert len(features) == len(results)

    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(
        'PrelimPrediction',
        ['feature_index', 'start_index', 'end_index', 'start_logit', 'end_logit']
    )

    all_predictions = []

    for example_index, example in enumerate(examples):
        cur_features = example_index_to_features[example_index]

        prelim_predictions = []
        for feature_index, feature in enumerate(cur_features):
            result = unique_id_to_result[feature.unique_id]
            n_best_start_index = _get_best_indexes(result.start_logits, n_best_size)
            n_best_end_index = _get_best_indexes(result.end_logits, n_best_size)

            for start_index in n_best_start_index:
                for end_index in n_best_end_index:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_origin_map:
                        continue
                    if end_index not in feature.token_to_origin_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue

                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]
                        )
                    )

            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True
            )

            _NbestPrediction = collections.namedtuple(
                'NbestPrediction', ['text', 'start_logit', 'end_logit']
            )

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break

                feature = cur_features[pred.feature_index]
                if pred.start_index > 0:
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(
                        tok_text, orig_text, do_lower_case=True, verbose=0)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True



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



if __name__ == '__main__':
    # 将 **上一步骤** 训练数据集的结果转为用于推断的 **原生数据**
    # convert_last_step_results_for_infer(
    #     results_path='../multi_turn_mrc_cls_task/results/for_infer/postprocessed/train_results.json',
    #     save_path='datasets/raw/for_infer/first_step/train.json',
    #     step='first'
    # )

    # 将 **上一步骤** 验证数据集的结果转为用于推断的 **原生数据**
    # convert_last_step_results_for_infer(
    #     results_path='../multi_turn_mrc_cls_task/results/for_infer/postprocessed/valid_results.json',
    #     save_path='datasets/raw/for_infer/first_step/valid.json',
    #     step='first'
    # )

    # # 为推断生成训练 tfrecord
    # generate_tfrecord_from_json_file(
    #     input_file_path='datasets/raw/for_infer/first_step/train.json',
    #     vocab_file_path='../vocabs/bert-base-chinese-vocab.txt',
    #     tfrecord_save_path='datasets/tfrecords/for_infer/first_step/train.tfrecord',
    #     meta_save_path='datasets/tfrecords/for_infer/first_step/train_meta.json',
    #     features_save_path='datasets/features/for_infer/first_step/train_features.pkl',
    #     max_seq_len=165,
    #     max_query_len=45,
    #     doc_stride=128,
    #     is_train=False
    # )
    #
    # # 为推断生成验证 tfrecord
    # generate_tfrecord_from_json_file(
    #     input_file_path='datasets/raw/for_infer/first_step/valid.json',
    #     vocab_file_path='../vocabs/bert-base-chinese-vocab.txt',
    #     tfrecord_save_path='datasets/tfrecords/for_infer/first_step/valid.tfrecord',
    #     meta_save_path='datasets/tfrecords/for_infer/first_step/valid_meta.json',
    #     features_save_path='datasets/features/for_infer/first_step/valid_features.pkl',
    #     max_seq_len=165,
    #     max_query_len=45,
    #     doc_stride=128,
    #     is_train=False
    # )

    # 将上一步骤训练数据的推断结果转为当前步骤的训练数据（含无答案问题）
    # convert_last_step_results_for_train(
    #     results_path='../multi_turn_mrc_cls_task/results/for_infer/postprocessed/train_results.json',
    #     save_path='datasets/raw/for_train/last_step/train.json'
    # )
    # 将上一步骤验证数据的推断结果转为当前步骤的验证数据（含无答案结果）
    # convert_last_step_results_for_train(
    #     results_path='../multi_turn_mrc_cls_task/results/for_infer/postprocessed/valid_results.json',
    #     save_path='datasets/raw/for_train/last_step/valid.json'
    # )

    # 处理 验证集的推断结果
    postprocess_results(
        raw_data_path='datasets/raw/for_infer/first_step/valid.json',
        features_path='datasets/features/for_infer/first_step/valid_features.pkl',
        results_path='results/for_infer/raw/first_step/valid_results.json',
        save_path='results/for_infer/postprocessed/first_step/valid_results.json'
    )

    pass
