# Copyright (c) 2019 NVIDIA CORPORATION.
#
# All rights reserved. Licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
#
# Many of the routines in this file are taken from:
# https://github.com/mlcommons/inference/tree/master/language/bert

from . import tokenization
from . import model

import collections
from dataclasses import dataclass
import json
import math
import tempfile
import transformers
import numpy as np
import string
import re


@dataclass
class SquadExample:
    question: str
    docTokens: list


class InputFeatures(object):
    """A single set of features of data. inputMask, segmentIds, and inputIds
    are passed separately because they must be passed directly to run()"""

    def __init__(self,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context):
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context


def load(dataPath):
    """Raw loading of input data with minimal preprocessing (runs on the client
    and is untimed).  Read a SQuAD json file into a list of (SquadExample, raw)
    where SquadExample is a parsed input to the preprocessor and raw is the raw
    dataset dictionary."""
    with open(dataPath, "r") as reader:
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
                question_text = qa["question"]

                example = SquadExample(question=question_text, docTokens=doc_tokens)
                examples.append((example, qa))

    return examples


def featurize(examples, vocab, cacheDir=None):
    """pre-process bert data. If cacheDir is provided, it will be used to read
    data (if available) or used to store cached pre-processed data.
    Returns: [(input_ids, input_mask, segment_ids, feature)] - one per example

    Note: The original bert code would return multiple features per example if
    the input text was long. We leave this part out for simplicity.
    """

    # This is an annoying limitation of the transformers library. It should
    # really accept an open file descriptor or bytes object so we can load from
    # binary data.
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(vocab)
        tokenizer = transformers.BertTokenizer(f.name)

    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length)

    return features


def interpret(startLogits, stopLogits, example, features):
    """Post-process the output of the bert model. start and stop logits are np
    arrays from the model output.
        start/stop Logits: Output of model (np.ndarray float32)
        example: SquadExample object used as input
        features: featurized version of example
    """
    # XXX I've hacked the preprocessor to only ever return one feature per
    # example. In theory, there could be more, but it complicates things too
    # much. I've left get_prediction() in its original form that handles
    # multiple features, but interpret() only takes one, hence making it a list
    # here.
    return get_prediction(startLogits, stopLogits, example, [features])


def get_final_text(pred_text, orig_text):
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
    #     pred_text = steve smith
    #     orig_text = Steve Smith's
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
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
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


def get_prediction(startLogits, endLogits, example, features):
    """Interpret the result of the bert model:
        start/stopLogits: output of model as np.ndarray float32
        example: the SquadExample object used in the prediction
        features: The featurized version of example
    """
    n_best_size = 20
    max_answer_length = 30

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    prelim_predictions = []

    for (feature_index, feature) in enumerate(features):
        start_indexes = _get_best_indexes(startLogits, n_best_size)
        end_indexes = _get_best_indexes(endLogits, n_best_size)

        # if we could have irrelevant answers, get the min score of irrelevant
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
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
                        start_logit=startLogits[start_index],
                        end_logit=endLogits[end_index]))

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
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.docTokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text)
        if final_text in seen_predictions:
            continue

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

    prediction = nbest_json[0]["text"]

    return prediction


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #       Question: What year was John Smith born?
    #       Context: The leader was John Smith (1895-1943).
    #       Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #       Question: What country is the top exporter of electornics?
    #       Context: The Japanese electronics industry is the lagest in the world.
    #       Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
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


max_seq_length = 384
max_query_length = 64
doc_stride = 128


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
    """Given a list of raw examples, return a list of feature lists for each
    example"""

    # list of lists of features per example
    features = []

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.docTokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        # XXX Technically, for long documents we need multiple features per
        # example. This complicates things for us, so I just skip it and only
        # do one span, no matter what
        # while start_offset < len(all_doc_tokens):
        while start_offset < 1:
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        # Per-example feature list
        exampleFeatures = []
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

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            feature = InputFeatures(
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context)

            # Run callback
            exampleFeatures.append((input_ids, input_mask, segment_ids, feature))

        # XXX See comment above at the doc_span calculation loop
        # features.append(exampleFeatures)
        features.append(exampleFeatures[0])

    return features


def normalize_answer(text):
    """Lower text and remove punctuation, articles and extra whitespace."""
    exclude = set(string.punctuation)

    text = text.lower()
    text = ''.join(ch for ch in text if ch not in exclude)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def check(pred, origData):
    """Validate prediction based on exact match. origData is the original entry
    in the dataset with all metadata, pred is the output of the
    post-processor."""

    groundTruths = list(map(lambda x: x['text'], origData['answers']))

    matches = []
    for truth in groundTruths:
        pred = normalize_answer(pred)
        truth = normalize_answer(truth)
        matches.append(pred == truth)

    return max(matches)


class bertModel(model.tvmModel):
    noPost = False
    preMap = model.inputMap(const=(0,), inp=(0,))
    runMap = model.inputMap(pre=(0, 1, 2))
    postMap = model.inputMap(inp=(0,), pre=(3,), run=(0, 1))

    nConst = 1
    nOutPre = 4
    nOutRun = 2
    nOutPost = 1

    @staticmethod
    def getConstants(modelDir):
        with open(modelDir / 'vocab.txt', 'rb') as f:
            vocab = f.read()
        return [vocab]

    @staticmethod
    def pre(inputs):
        vocab = inputs[0]
        example = inputs[1]

        # featurize() can handle batches, but we only support batch size 1 right
        # now
        inputIds, inputMask, segmentIds, otherFeature = featurize([example], vocab)[0]
        inputIds = np.array(inputIds).astype(np.int64)[np.newaxis, :].tobytes()
        inputMask = np.array(inputMask).astype(np.int64)[np.newaxis, :].tobytes()
        segmentIds = np.array(segmentIds).astype(np.int64)[np.newaxis, :].tobytes()
        return [inputIds, inputMask, segmentIds, otherFeature]

    @staticmethod
    def post(inputs):
        example = inputs[0]
        feature = inputs[1]
        startLogits = inputs[2]
        endLogits = inputs[3]

        startLogits = np.frombuffer(startLogits, dtype=np.float32).tolist()
        endLogits = np.frombuffer(endLogits, dtype=np.float32).tolist()

        pred = interpret(startLogits, endLogits, example, feature)
        return pred

    @staticmethod
    def getMlPerfCfg(testing=False):
        settings = model.getDefaultMlPerfCfg()

        settings.server_target_qps = 0.3
        # if testing:
        #     # settings.server_target_qps = 0.3
        #     settings.server_target_latency_ns = 1000
        # else:
        #     settings.server_target_latency_ns = 1000000000

        return settings
