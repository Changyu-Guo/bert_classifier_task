# -*- coding: utf - 8 -*-

import os
import copy
import json
import math
import collections

import six
import tensorflow as tf
from absl import logging
import tokenization


def write_predictions(all_examples,
                      all_features,
                      all_results,
                      n_best_size,
                      max_answer_length,
                      do_lower_case,
                      output_prediction_file,
                      output_nbest_file,
                      output_null_log_odds_file,
                      version_2_with_negative=False,
                      verbose=False):
    logging.info("Writing predictions to: %s", output_prediction_file)
    logging.info("Writing nbest to: %s", output_nbest_file)

    all_predictions, all_nbest_json, only_text_predictions = (
        postprocess_output(
            all_examples=all_examples,
            all_features=all_features,
            all_results=all_results,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            do_lower_case=do_lower_case,
            verbose=verbose))

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
        verbose=False
):
    """Postprocess model output, to form prediction results."""

    # 每个 example 可能对应多个 feature
    # example_index_to_features 的作用是根据 example 的 index 找到其对应的 features
    # 在 convert_examples_to_features 的时候，每个 feature 都保存了其对应的 example_index
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    # 每个 feature 都有一个 unique_id, 可以根据 unique_id 确定一个 feature 对应的答案
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    # 候选答案
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    only_answer_predictions = collections.OrderedDict()
    all_predictions = []
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
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

        # start_logit + end_logit 越大认为这个答案越好
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True
        )

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
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

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
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

        # TODO 此处只输出了预测答案，应在此处加上 原始文章、原始问题、原始答案
        pred_dict_item = {
            "origin_text_tokens": example.doc_tokens,
            "question": example.question_text,
            "correct_answer": example.orig_answer_text,
            "pred_answer": nbest_json[0]['text']
        }

        only_answer_predictions[example.qas_id] = nbest_json[0]['text']
        all_predictions.append(pred_dict_item)

        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json, only_answer_predictions


def write_to_json_files(json_records, json_file):
    with tf.io.gfile.GFile(json_file, "w") as writer:
        writer.write(json.dumps(json_records, indent=2, ensure_ascii=False) + "\n")


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


def generate_train_tf_record_from_json_file(
        input_file_path,
        vocab_file_path,
        output_path,
        max_seq_length=384,
        do_lower_case=True,
        max_query_length=64,
        doc_stride=128,
        version_2_with_negative=False
):
    """Generates and saves training data into a tf record file."""
    train_examples = read_squad_examples(
        input_file=input_file_path,
        is_training=True,
        version_2_with_negative=version_2_with_negative)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file_path, do_lower_case=do_lower_case)
    train_writer = FeatureWriter(filename=output_path, is_training=True)
    number_of_examples = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature)
    train_writer.close()

    meta_data = {
        "task_type": "bert_squad_train",
        "train_data_size": number_of_examples,
        "max_seq_len": max_seq_length,
        "max_query_len": max_query_length,
        "doc_stride": doc_stride,
        "version_2_with_negative": version_2_with_negative,
    }

    return meta_data


def generate_valid_tf_record_from_json_file(
        input_file_path,
        vocab_file_path,
        output_path,
        max_seq_length=384,
        do_lower_case=True,
        max_query_length=64,
        doc_stride=128,
        version_2_with_negative=False,
        batch_size=None,
        is_training=True
):
    valid_examples = read_squad_examples(
        input_file=input_file_path,
        is_training=is_training,
        version_2_with_negative=version_2_with_negative)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file_path, do_lower_case=do_lower_case)
    valid_writer = FeatureWriter(filename=output_path, is_training=is_training)

    valid_features = []

    def _append_feature(feature, is_padding):
        if not is_padding:
            valid_features.append(feature)
        valid_writer.process_feature(feature)

    if is_training:
        output_fn = valid_writer.process_feature
    else:
        output_fn = _append_feature

    number_of_examples = convert_examples_to_features(
        examples=valid_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training,
        output_fn=output_fn,
        batch_size=batch_size
    )
    valid_writer.close()

    meta_data = {
        "task_type": "bert_squad_valid",
        "valid_data_size": number_of_examples,
        "max_seq_len": max_seq_length,
        "max_query_len": max_query_length,
        "doc_stride": doc_stride,
        "version_2_with_negative": version_2_with_negative,
    }

    return meta_data


if __name__ == '__main__':
    pass
