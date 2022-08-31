# Copyright 2022 The KerasNLP Authors
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
"""Tests for Bert model."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models import bert


class BertTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.model = bert.BertCustom(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            name="encoder",
        )
        self.batch_size = 2
        self.input_batch = {
            "token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "segment_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).repeat(3)

    def test_valid_call_bert(self):
        self.model(self.input_batch)

    def test_variable_sequence_length_call_bert(self):
        for seq_length in (25, 50, 75):
            input_data = {
                "token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "segment_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
            }
            self.model(input_data)

    def test_valid_call_classifier(self):
        classifier = bert.BertClassifier(self.model, 4, name="classifier")
        classifier(self.input_batch)

    def test_valid_call_bert_base(self):
        model = bert.BertBase(vocabulary_size=1000, name="encoder")
        input_data = {
            "token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "segment_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }
        model(input_data)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_base_compile(self, jit_compile):
        model = bert.BertBase(vocabulary_size=1000, name="encoder")
        model.compile(jit_compile=jit_compile)
        model.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_base_compile_batched_ds(self, jit_compile):
        model = bert.BertBase(vocabulary_size=1000, name="encoder")
        model.compile(jit_compile=jit_compile)
        model.predict(self.input_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_classifier_compile(self, jit_compile):
        model = bert.BertClassifier(self.model, 4, name="classifier")
        model.compile(jit_compile=jit_compile)
        model.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_classifier_compile_batched_ds(self, jit_compile):
        model = bert.BertClassifier(self.model, 4, name="classifier")
        model.compile(jit_compile=jit_compile)
        model.predict(self.input_dataset)

    def test_bert_base_vocab_error(self):
        # Need `vocabulary_size` or `weights`
        with self.assertRaises(ValueError):
            bert.BertBase(name="encoder")

        # Only one of `vocabulary_size` or `weights`
        with self.assertRaises(ValueError):
            bert.BertBase(
                weights="bert_base_uncased",
                vocabulary_size=1000,
                name="encoder",
            )

        # Not a checkpoint name
        with self.assertRaises(ValueError):
            bert.BertBase(
                weights="bert_base_clowntown",
                name="encoder",
            )

    def test_saving_model(self):
        model_output = self.model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), "model")
        self.model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            model_output["pooled_output"], restored_output["pooled_output"]
        )
