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

"""ROUGE metric implementation based on `keras.metrics.Metric`."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.utils.tensor_utils import tensor_to_string_list

try:
    from rouge_score import rouge_scorer
except:
    pass


class Rouge(keras.metrics.Metric):
    """ROUGE metric.

    This class implements all the variants of the ROUGE metric - ROUGE-N,
    ROUGE-L and ROUGE-LSum.

    Args:
        variant: string. One of "rougeN", "rougeL", "rougeLsum". Defaults to
            "rouge2". For "rougeN", N lies in the range [1, 9].
        metric_type: string. One of "precision", "recall", "f1_score". Defaults
            to "f1_score".
        use_stemmer: bool. Whether Porter Stemmer should be used to strip word
            suffixes to improve matching. Defaults to False.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to tf.float32.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.
    """

    def __init__(
        self,
        variant="rouge2",
        metric_type="f1_score",
        use_stemmer=False,
        dtype=None,
        name="rouge",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if rouge_scorer is None:
            raise ImportError(
                "ROUGE metric requires the `rouge_score` package."
                "Please install it with `pip install rouge_score`."
            )

        if not tf.as_dtype(self.dtype).is_floating:
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        if variant not in tuple(
            ("rouge" + str(order) for order in range(1, 10))
        ) + (
            "rougeL",
            "rougeLsum",
        ):
            raise ValueError(
                "Invalid variant of ROUGE. Should be one of: rougeN, rougeL, "
                "rougeLsum, with N ranging from 1 to 9. Received: "
                f"variant={variant}"
            )
        if metric_type not in ("precision", "recall", "f1_score"):
            raise ValueError(
                '`metric_type` must be one of "precision", "recall", '
                f'"f1_score". Received: metric_type={metric_type}'
            )

        self.variant = variant
        self.metric_type = metric_type
        self.use_stemmer = use_stemmer

        # To-do: Add split_summaries and tokenizer options after the maintainers
        # of rouge_scorer have released a new version.
        self._rouge_scorer = rouge_scorer.RougeScorer(
            rouge_types=[self.variant],
            use_stemmer=use_stemmer,
        )

        self._rouge_score = self.add_weight(
            name="rouge_score",
            initializer="zeros",
            dtype=self.dtype,
        )
        self._number_of_samples = self.add_weight(
            name="number_of_samples", initializer="zeros", dtype=self.dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Both y_true and y_pred have shape: [batch_size]. Each element is a
        # string.

        # Check if input is a raw string/list.
        if isinstance(y_true, str):
            y_true = tf.constant([y_true])
        elif isinstance(y_true, list):
            y_true = tf.constant(y_true)
        if isinstance(y_pred, str):
            y_pred = tf.constant([y_pred])
        elif isinstance(y_pred, list):
            y_pred = tf.constant(y_pred)

        batch_size = tf.shape(y_true)[0]

        def _calculate_rouge_score(reference, hypothesis):
            reference = tensor_to_string_list(reference)
            hypothesis = tensor_to_string_list(hypothesis)
            score = self._rouge_scorer.score(reference, hypothesis)[
                self.variant
            ]

            if self.metric_type == "precision":
                score = score.precision
            elif self.metric_type == "recall":
                score = score.recall
            else:
                score = score.fmeasure
            return score

        for batch_idx in range(batch_size):
            score = tf.py_function(
                func=_calculate_rouge_score,
                inp=[y_true[batch_idx], y_pred[batch_idx]],
                Tout=self.dtype,
            )
            self._rouge_score.assign_add(score)

        self._number_of_samples.assign_add(
            tf.cast(batch_size, dtype=self.dtype)
        )

    def result(self):
        if self._number_of_samples == 0:
            return 0.0
        rouge_l_score = self._rouge_score / self._number_of_samples
        return rouge_l_score

    def reset_state(self):
        self._rouge_score.assign(0.0)
        self._number_of_samples.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "metric_type": self.metric_type,
                "use_stemmer": self.use_stemmer,
            }
        )
        return config
