# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base

    Convert MRC data to squad format.
"""

import json
import collections
import tensorflow as tf
import tokenization
from data_processors.commom import read_init_train_train_examples
from data_processors.commom import read_init_train_valid_examples
from data_processors.commom import get_squad_json_template
from data_processors.commom import get_squad_json_paragraph_template
from data_processors.commom import get_squad_json_qas_item_template
from data_processors.commom import extract_examples_dict_from_relation_questions

# mrc task
MRC_TRAIN_SAVE_PATH = 'common-datasets/preprocessed_datasets/mrc_train.json'
MRC_VALID_SAVE_PATH = 'common-datasets/preprocessed_datasets/mrc_valid.json'

# multi label cls step inference result
multi_label_cls_train_results_path = 'inference_results/multi_label_cls_results/in_use/train_results.json'
multi_label_cls_valid_results_path = 'inference_results/multi_label_cls_results/in_use/valid_results.json'

# bi cls s1 inference result
bi_cls_s1_train_results_path = 'inference_results/bi_cls_s1_results/train_results.json'
bi_cls_s1_valid_results_path = 'inference_results/bi_cls_s1_results/valid_results.json'

# first step json file
train_data_before_first_step_save_path = 'common-datasets/preprocessed_datasets/first_step_train.json'
valid_data_before_first_step_save_path = 'common-datasets/preprocessed_datasets/first_step_valid.json'

# first step inference inference_results
first_step_inference_train_save_path = 'inference_results/mrc_results/in_use/first/train_results.json'
first_step_inference_valid_save_path = 'inference_results/mrc_results/in_use/first/valid_results.json'

# second step json file
second_step_train_save_path = 'common-datasets/preprocessed_datasets/second_step_train.json'
second_step_valid_save_path = 'common-datasets/preprocessed_datasets/second_step_valid.json'

# second step inference inference_results
second_step_inference_train_save_path = 'inference_results/mrc_results/in_use/second/train_results.json'
second_step_inference_valid_save_path = 'inference_results/mrc_results/in_use/second/valid_results.json'


def convert_init_train_examples_to_squad_json_format(output_save_path, is_train=True):
    if is_train:
        examples = read_init_train_train_examples()
        squad_json_title = 'train examples'
    else:
        squad_json_title = 'valid examples'
        examples = read_init_train_valid_examples()

    relation_questions_dict = extract_examples_dict_from_relation_questions()

    squad_json = get_squad_json_template(squad_json_title)

    qas_id = 1
    for example_index, example in enumerate(examples):
        text = example['text']
        relations = example['relations']

        # 因为在构造某一个新样本的时候会丢失一部分的信息
        # 因此在此处保存完整的样本信息，以便于在推断任务中使用
        squad_json_paragraph = get_squad_json_paragraph_template(
            text=text,
            origin_relations=relations
        )

        # 一个文本对应多个 relation
        # 一个 relation 对应一个问答对
        for relation in relations:
            s = relation['subject']
            r = relation['relation']
            o = relation['object']

            s_start_pos = text.find(s)
            o_start_pos = text.find(o)

            relation_question = relation_questions_dict[r]
            relation_question_a = relation_question.relation_question_a
            relation_question_b = relation_question.relation_question_b.replace('subject', s)

            squad_json_qas_item_a = get_squad_json_qas_item_template(
                question=relation_question_a,
                answers=[{
                    'text': s,
                    'answer_start': s_start_pos
                }],
                qas_id='id_' + str(qas_id)
            )
            qas_id += 1
            squad_json_qas_item_b = get_squad_json_qas_item_template(
                question=relation_question_b,
                answers=[{
                    'text': o,
                    'answer_start': o_start_pos
                }],
                qas_id='id_' + str(qas_id)
            )
            qas_id += 1

            squad_json_paragraph['qas'].append(squad_json_qas_item_a)
            squad_json_paragraph['qas'].append(squad_json_qas_item_b)

        squad_json['data'][0]['paragraphs'].append(squad_json_paragraph)

    with tf.io.gfile.GFile(output_save_path, mode='w') as writer:
        writer.write(json.dumps(squad_json, ensure_ascii=False, indent=2))
    writer.close()


class SquadExample:

    def __init__(
            self,
            qas_id,
            question_text,
            doc_tokens,
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


class InputFeatures:

    def __init__(
            self,
            unique_id,
            example_index,
            doc_span_index,
            tokens,
            token_to_orig_map,
            token_is_max_context,
            input_ids,
            input_mask,
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
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class FeatureWriter(object):

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
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
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
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
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

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
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
                is_impossible=example.is_impossible)

            # Run callback
            if is_training:
                output_fn(feature)
            else:
                output_fn(feature, is_padding=False)

            unique_id += 1

    if not is_training and feature:
        assert batch_size
        num_padding = 0
        num_examples = unique_id - base_id
        if unique_id % batch_size != 0:
            num_padding = batch_size - (num_examples % batch_size)
        logging.info("Adding padding examples to make sure no partial batch.")
        logging.info("Adds %d padding examples for inference.", num_padding)
        dummy_feature = copy.deepcopy(feature)
        for _ in range(num_padding):
            dummy_feature.unique_id = unique_id

            # Run callback
            output_fn(feature, is_padding=True)
            unique_id += 1

    return unique_id - base_id


def generate_tfrecord_from_json_file(
        input_file_path,
        vocab_file_path,
        output_file_path,
        max_seq_len=384,
        max_query_len=64,
        doc_stride=128
):
    examples = read_squad_examples(
        input_file=input_file_path,
        is_training=is_training
    )

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path)
    writer = FeatureWriter(filename=output_file_path, is_training=True)
    convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        doc_stride=doc_stride,
        max_query_length=max_query_len,
        is_training=True,
        output_fn=writer.process_feature,
        batch_size=None
    )

    meta_data = {
        'data_size': writer.num_features,
        'max_seq_len': max_seq_len,
        'max_query_len': max_query_len,
        'doc_stride': doc_stride
    }

    return meta_data


def convert_inference_results_for_second_step(inference_results_path, convert_results_save_path):
    relation_questions_dict = extract_examples_from_relation_questions()
    with tf.io.gfile.GFile(inference_results_path, mode='r') as reader:
        input_data = json.load(reader)['data']
    reader.close()

    _id = 0
    for i in range(len(input_data[0]['paragraphs'])):
        item = input_data[0]['paragraphs'][i]
        pred_relations = item['pred_relations']

        qas = []

        for relation in pred_relations:
            _subject = relation['subject']
            relation_question = relation_questions_dict[relation['relation']]
            relation_question_a = relation_question.relation_question_b.replace('subject', _subject)

            qas_item = {
                'question': relation_question_a,
                'relation': relation['relation'],
                'subject': _subject,
                'id': 'id_' + str(_id)
            }
            qas.append(qas_item)
            _id += 1

        input_data[0]['paragraphs'][i]['qas'] = qas

        if (i + 1) % 1000 == 0:
            print(i + 1)

    input_data = {
        'data': input_data
    }
    with tf.io.gfile.GFile(convert_results_save_path, mode='w') as writer:
        writer.write(json.dumps(input_data, ensure_ascii=False, indent=2))
    writer.close()


def mrc_data_processor_main():
    convert_inference_results_for_second_step(
        inference_results_path=first_step_inference_train_save_path,
        convert_results_save_path=second_step_train_save_path
    )
    # convert_inference_results_for_second_step(
    #     inference_results_path=first_step_inference_valid_save_path,
    #     convert_results_save_path=second_step_valid_save_path
    # )


if __name__ == '__main__':
    pass
