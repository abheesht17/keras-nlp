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
import tensorflow as tf

from keras_nlp.models.task import Task


class WhisperCausalLM(Task):
    """An end-to-end Whisper encoder-decoder model for transcribing audio.

    The output logits layer is added on top of `WhisperBackbone`. This output
    logits layer is the transposed decoder embedding layer, that is, the shape
    is `(hidden_size, vocabulary_size)`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/openai/whisper).

    Args:
        backbone: A `WhisperBackbone` instance.
    """

    def __init__(
        self,
        backbone,
        **kwargs,
    ):
        inputs = backbone.input
        x = backbone(inputs)["decoder_sequence_output"]
        # Use token embedding weights to project from the token representation
        # to vocabulary logits.
        outputs = tf.matmul(
            x,
            backbone.decoder_token_embedding.embeddings,
            transpose_b=True,
        )

        # Instantiate using Functional API Model constructor.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=False,
            **kwargs,
        )

        self.backbone = backbone
