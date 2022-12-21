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
"""Tests for loading pretrained model presets."""

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.models.deberta.deberta_backbone import DebertaBackbone
from keras_nlp.models.deberta.deberta_classifier import DebertaClassifier
from keras_nlp.models.deberta.deberta_preprocessor import DebertaPreprocessor
from keras_nlp.models.deberta.deberta_tokenizer import DebertaTokenizer


@pytest.mark.large
class DebertaPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for DeBERTa presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/deberta/deberta_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = DebertaTokenizer.from_preset(
            "deberta_xsmall",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [581, 63773, 119455, 6, 147797, 5]
        self.assertAllEqual(outputs, expected_outputs)

    def test_preprocessor_output(self):
        preprocessor = DebertaPreprocessor.from_preset(
            "deberta_xsmall",
            sequence_length=4,
        )
        outputs = preprocessor("The quick brown fox.")["token_ids"]
        expected_outputs = [0, 581, 63773, 2]
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_backbone_output(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[0, 581, 63773, 2]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = DebertaBackbone.from_preset(
            "deberta_xsmall", load_weights=load_weights
        )
        outputs = model(input_data)
        if load_weights:
            outputs = outputs[0, 0, :5]
            expected = [0.084763, 0.097018, 0.051329, -0.000805, 0.028415]
            self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_classifier_output(self, load_weights):
        input_data = tf.constant(["The quick brown fox."])
        model = DebertaClassifier.from_preset(
            "deberta_xsmall", load_weights=load_weights
        )
        # Never assert output values, as the head weights are random.
        model.predict(input_data)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_classifier_output_without_preprocessing(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[0, 581, 63773, 2]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = DebertaClassifier.from_preset(
            "deberta_xsmall",
            load_weights=load_weights,
            preprocessor=None,
        )
        # Never assert output values, as the head weights are random.
        model.predict(input_data)

    @parameterized.named_parameters(
        ("deberta_tokenizer", DebertaTokenizer),
        ("deberta_preprocessor", DebertaPreprocessor),
        ("deberta", DebertaBackbone),
        ("deberta_classifier", DebertaClassifier),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("deberta_tokenizer", DebertaTokenizer),
        ("deberta_preprocessor", DebertaPreprocessor),
        ("deberta", DebertaBackbone),
        ("deberta_classifier", DebertaClassifier),
    )
    def test_unknown_preset_error(self, cls):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("deberta_xsmall_clowntown")


@pytest.mark.extra_large
class DebertaPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.

    This tests every DeBERTa preset and is only run manually.
    Run with:
    `pytest keras_nlp/models/deberta/deberta_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_deberta(self, load_weights):
        for preset in DebertaBackbone.presets:
            model = DebertaBackbone.from_preset(
                preset, load_weights=load_weights
            )
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            model(input_data)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_deberta_classifier(self, load_weights):
        for preset in DebertaClassifier.presets:
            classifier = DebertaClassifier.from_preset(
                preset,
                num_classes=4,
                load_weights=load_weights,
            )
            input_data = tf.constant(["This quick brown fox"])
            classifier.predict(input_data)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_deberta_classifier_without_preprocessing(self, load_weights):
        for preset in DebertaClassifier.presets:
            classifier = DebertaClassifier.from_preset(
                preset,
                num_classes=4,
                load_weights=load_weights,
                preprocessor=None,
            )
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512),
                    dtype=tf.int64,
                    maxval=classifier.backbone.vocabulary_size,
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            classifier.predict(input_data)

    def test_load_tokenizers(self):
        for preset in DebertaTokenizer.presets:
            tokenizer = DebertaTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_preprocessors(self):
        for preset in DebertaPreprocessor.presets:
            preprocessor = DebertaPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
