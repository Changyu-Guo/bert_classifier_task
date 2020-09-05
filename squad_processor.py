# -*- coding: utf - 8 -*-

import os
import copy
import json
import math
import collections

import six
import tensorflow as tf
from absl import logging
import tokenization as bert_tokenization


class SquadExample(object):
    def __init__(
            self,
            qas_id,
            question_text,
            doc_tokens,  # white space tokenize
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=False
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (bert_tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            bert_tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.start_position:
            s += ", is_impossible: %r" % self.is_impossible
        return s


class InputFeatures(object):
    def __init__(
            self,
            unique_id,
            example_index,
            doc_span_index,
            tokens,  # word piece tokenize
            token_to_orig_map,  # word piece token -> white space token
            token_is_max_context,
            input_ids,
            input_mask,
            segment_ids,
            start_position=None,  # index in input_ids
            end_position=None,  # index in input_ids
            is_impossible=None
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class FeatureWriter:
    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        tf.io.gfile.makedirs(os.path.dirname(filename))
        self._writer = tf.io.TFRecordWriter(filename)

    def process_feature(self, feature):
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["inputs_ids"] = create_int_feature(feature.input_ids)
        features["inputs_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_position"] = create_int_feature([feature.start_position])
            features["end_position"] = create_int_feature([feature.end_position])
            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def read_squad_examples(input_file, is_training, version_2_with_negative):
    with tf.io.gfile.GFile(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:

            # 对于每一个 context
            # 都要将其转换为 doc_tokens
            # 以便于后面构建 start_position 和 end_position
            # 在转换的过程中要记录 context 中每个字符的位置对应到 doc_tokens 中的 index
            # 由于原始数据集中提供的答案位置是在上下文中的字符位置
            # 则可以根据记录到的对应位置确定答案在 doc_tokens 中的位置
            # 即可以得到 start_position 和 end_position
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

                # 不管是不是空格都应该 append
                # 以保证对应位置不会出错
                char_to_word_offset.append(len(doc_tokens) - 1)

            # 抽取当前上下文对应的问题和答案
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:

                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)

                        # 根据之前的记录确认答案在 doc_tokens 中的起始位置
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]

                        # doc_tokens 中的字符组成的答案
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])

                        # 真正的答案
                        cleaned_answer_text = " ".join(
                            bert_tokenization.whitespace_tokenize(orig_answer_text))

                        # 如果答案不包含在 doc_tokens 中的答案中
                        # 则在训练的过程中一定会存在偏差
                        # 则舍弃这一条数据
                        if actual_text.find(cleaned_answer_text) == -1:
                            logging.warning("Could not find answer: '%s' vs. '%s'",
                                            actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)

    return examples


def convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        output_fn
):
    base_id = 1000000000
    unique_id = base_id
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # 使用 word piece 进行更细粒度的分词
        # 因为仅仅使用 white space 分词得到的答案不精确
        tok_to_orig_index = []  # tokens -> doc_tokens
        orig_to_tok_index = []  # doc_tokens -> tokens (只能对应到 tokens 中 token 的开始位置)
        all_doc_tokens = []  # word piece 分词得到的 tokens
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # 确定答案在 tokens 中的 position
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

        # [CLS], [SEP], [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"])

        # 使用开始位置和 span 的长度记录了所有的 span
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset

            # 截断
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        # 将问题和每一个上下文片段构造成一个样本
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

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # 确定答案在 doc_span 中的位置
            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
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

            if example_index < 2:
                logging.info("*** Example ***")
                logging.info("unique_id: %s", unique_id)
                logging.info("example_index: %s", example_index)
                logging.info("doc_span_index: %s", doc_span_index)
                logging.info("tokens: %s",
                             " ".join([bert_tokenization.printable_text(x) for x in tokens]))
                logging.info(
                    "token_to_orig_map: %s", " ".join([
                        "%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)
                    ]))
                logging.info(
                    "token_is_max_context: %s", " ".join([
                        "%d:%s" % (x, y)
                        for (x, y) in six.iteritems(token_is_max_context)
                    ]))
                logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logging.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logging.info("start_position: %d", start_position)
                    logging.info("end_position: %d", end_position)
                    logging.info("answer: %s", bert_tokenization.printable_text(answer_text))

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible
            )

            # Run callback
            output_fn(feature)

            unique_id += 1

        if (example_index + 1) % 1000 == 0:
            print(example_index + 1)

    return unique_id - base_id


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


def write_predictions(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        verbose=False
):
    logging.info("Writing predictions to: %s", output_prediction_file)
    logging.info("Writing nbest to: %s", output_nbest_file)

    all_predictions, all_nbest_json, scores_diff_json = (
        postprocess_output(
          all_examples=all_examples,
          all_features=all_features,
          all_results=all_results,
          n_best_size=n_best_size,
          max_answer_length=max_answer_length,
          do_lower_case=do_lower_case,
          version_2_with_negative=version_2_with_negative,
          null_score_diff_threshold=null_score_diff_threshold,
          verbose=verbose
        )
    )

    write_to_json_files(all_predictions, output_prediction_file)
    write_to_json_files(all_nbest_json, output_nbest_file)
    if version_2_with_negative:
        write_to_json_files(scores_diff_json, output_null_log_odds_file)


def postprocess_output(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        verbose=False
):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        score_null = 1000000
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
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
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
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
                    tok_text, orig_text, do_lower_case, verbose=verbose)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit))
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

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
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            if best_non_null_entry is not None:
                score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            else:
                logging.warning("best_non_null_entry is None")
                scores_diff_json[example.qas_id] = score_null
                all_predictions[example.qas_id] = ""

        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json, scores_diff_json


def write_to_json_files(json_records, json_file):
    with tf.io.gfile.GFile(json_file, "w") as writer:
        writer.write(json.dumps(json_records, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose=False):
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

    tokenizer = bert_tokenization.BasicTokenizer(do_lower_case=do_lower_case)

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
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
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


def save_squad_dataset(dataset, path):
    writer = FeatureWriter(path, is_training=True)
    _id = 0
    for data in dataset:
        feature = InputFeatures(
            unique_id=_id,
            example_index=None,
            doc_span_index=None,
            tokens=None,
            token_to_orig_map=None,
            token_is_max_context=None,
            input_ids=data[0][0].numpy(),
            input_mask=data[0][1].numpy(),
            segment_ids=data[0][2].numpy(),
            start_position=data[1].numpy(),
            end_position=data[2].numpy(),
            is_impossible=False
        )
        writer.process_feature(feature)

    writer.close()


def generate_tf_record_from_json_file(
        train_file_path,
        eval_file_path,
        vocab_file_path,
        output_dir,
        output_prefix='',
        max_seq_length=384,
        do_lower_case=True,
        max_query_length=64,
        doc_stride=128,
        version_2_with_negative=False
):
    train_file_output_path = os.path.join(output_dir, output_prefix + '_' + 'train.tfrecord')
    eval_file_output_path = os.path.join(output_dir, output_prefix + '_' + 'valid.tfrecord')
    train_meta_data_output_path = os.path.join(output_dir, output_prefix + '_' + 'train_meta_data.json')
    eval_meta_data_output_path = os.path.join(output_dir, output_prefix + '_' + 'valid_meta_data.json')
    train_examples = read_squad_examples(
        input_file=train_file_path,
        is_training=True,
        version_2_with_negative=version_2_with_negative
    )
    eval_examples = read_squad_examples(
        input_file=eval_file_path,
        is_training=True,
        version_2_with_negative=version_2_with_negative
    )

    tokenizer = bert_tokenization.FullTokenizer(
        vocab_file=vocab_file_path,
        do_lower_case=do_lower_case
    )

    train_writer = FeatureWriter(filename=train_file_output_path, is_training=True)
    eval_writer = FeatureWriter(filename=eval_file_output_path, is_training=False)

    number_of_train_examples = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature
    )
    train_writer.close()

    number_of_eval_examples = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True,
        output_fn=eval_writer.process_feature
    )
    eval_writer.close()

    train_meta_data = {
        "task_type": "bert_squad_train",
        "train_data_size": number_of_train_examples,
        "max_seq_length": max_seq_length,
        "max_query_length": max_query_length,
        "doc_stride": doc_stride,
        "version_2_with_negative": version_2_with_negative,
    }
    eval_meta_data = {
        "task_type": "bert_squad_eval",
        "eval_data_size": number_of_eval_examples,
        "max_seq_length": max_seq_length,
        "max_query_length": max_query_length,
        "doc_stride": doc_stride,
        "version_2_with_negative": version_2_with_negative,
    }

    with open(train_meta_data_output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_meta_data, ensure_ascii=False, indent=2))
    f.close()

    with open(eval_meta_data_output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(eval_meta_data, ensure_ascii=True, indent=2))
    f.close()

    return train_meta_data, eval_meta_data


if __name__ == '__main__':
    generate_tf_record_from_json_file(
        train_file_path='datasets/preprocessed_datasets/qa_train_examples.json',
        eval_file_path='datasets/preprocessed_datasets/qa_valid_examples.json',
        vocab_file_path='./vocab.txt',
        output_dir='./datasets/tfrecord_datasets',
        output_prefix='mrc',
        max_seq_length=248,
        do_lower_case=True,
        max_query_length=32,
        doc_stride=128,
        version_2_with_negative=False
    )
