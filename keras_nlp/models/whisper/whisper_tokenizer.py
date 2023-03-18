# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whisper tokenizer."""

import json

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.whisper.whisper_presets import LANGUAGE_TOKENS
from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer


def _load_dict(dict_or_path):
    if isinstance(dict_or_path, str):
        with open(dict_or_path, "r") as f:
            dict_or_path = json.load(f)
    return dict_or_path


@keras_nlp_export("keras_nlp.models.WhisperTokenizer")
class WhisperTokenizer(BytePairTokenizer):
    """Whisper text tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.BytePairTokenizer`.
    This tokenizer does not provide truncation or padding of inputs.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.
        special_tokens: string or dict, maps special tokens to integer IDs. If
            it is a string, it should be the path to a JSON file.
        is_multilingual: bool, whether the tokenizer is multilingual.
    """

    def __init__(
        self,
        vocabulary,
        merges,
        special_tokens,
        is_multilingual,
        **kwargs,
    ):
        vocabulary = _load_dict(vocabulary)

        # Necessary special tokens.
        bos_token = "<|startoftranscript|>"
        eos_token = "<|endoftext|>"

        if is_multilingual:
            # Multilingual tokenizer.
            pad_token = ""

            # Add language tokens to the vocabulary. This makes detokenization
            # easier for us.
            vocabulary = {
                **vocabulary,
                **LANGUAGE_TOKENS,
            }
        else:
            # English tokenizer.
            pad_token = "<|endoftext|>"

        no_timestamps_token = "<|notimestamps|>"
        # Task special tokens.
        translate_token = "<|translate|>"
        transcribe_token = "<|transcribe|>"

        special_tokens = _load_dict(special_tokens)
        for token in [
            bos_token,
            eos_token,
            pad_token,
            no_timestamps_token,
            translate_token,
            transcribe_token,
        ]:
            if token not in special_tokens:
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`special_tokens`. Please provide `'{token}'` in your "
                    "`special_tokens`."
                )
            # Add special tokens to `vocabulary` for easy detokenization.
            vocabulary[token] = special_tokens[token]

        self.bos_token_id = special_tokens[bos_token]
        self.eos_token_id = special_tokens[eos_token]
        self.pad_token_id = special_tokens[pad_token]
        self.no_timestamps_token_id = special_tokens[no_timestamps_token]
        self.translate_token_id = special_tokens[translate_token]
        self.transcribe_token_id = special_tokens[transcribe_token]

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        self.is_multilingual = is_multilingual
