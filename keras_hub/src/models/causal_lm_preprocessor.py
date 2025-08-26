import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.CausalLMPreprocessor")
class CausalLMPreprocessor(Preprocessor):
    """Base class for causal language modeling preprocessing layers.

    `CausalLMPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
    create a preprocessing layer for causal language modeling tasks. It is
    intended to be paired with a `keras.models.CausalLM` task.

    All `CausalLMPreprocessor` take inputs a single input. This can be a single
    string or a batch of strings. See examples below. These inputs
    will be tokenized and padded/truncated to a fixed sequence length.

    This layer will always output a `(x, y, sample_weight)` tuple, where `x`
    is a dictionary with the tokenized inputs, `y` contains the tokens from `x`
    offset by 1, and `sample_weight` marks where `y` contains padded
    values. The exact contents of `x` will vary depending on the model being
    used.

    a `CausalLMPreprocessor` contains two extra methods, `generate_preprocess`
    and `generate_postprocess` for use with generation. See examples below.

    All `CausalLMPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained config and vocabularies. You can
    call the `from_preset()` constructor directly on this base class, in which
    case the correct class for you model will be automatically instantiated.

    Examples.
    ```python
    preprocessor = keras_hub.models.CausalLMPreprocessor.from_preset(
        "bert_base_en_uncased",
        sequence_length=256, # Optional.
    )

    # Tokenize, mask and pack a single sentence.
    x = "The quick brown fox jumped."
    x, y, sample_weight = preprocessor(x)

    # Tokenize and pad/truncate a batch of labeled sentences.
    x = ["The quick brown fox jumped.", "Call me Ishmael."]
    x, y, sample_weight = preprocessor(x)

    # With a `tf.data.Dataset`.
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Generate preprocess and postprocess.
    x = preprocessor.generate_preprocess(x)  # Tokenized numeric inputs.
    x = preprocessor.generate_postprocess(x)  # Detokenized string outputs.
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        max_prompt_length=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = None
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.max_prompt_length = max_prompt_length

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = MultiSegmentPacker(
            sequence_length=self.sequence_length,
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            sep_value=[],
            pad_value=self.tokenizer.pad_token_id,
        )
        self.built = True

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        x = self.tokenizer(x)
        # Pad with one extra token to account for the truncation below.
        token_ids, _ = self.packer(
            x,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }
        # Target `y` will be the next token.
        y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def dpo_preprocess(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
        max_prompt_length=None,
    ):
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        max_prompt_length = max_prompt_length or self.max_prompt_length
        if max_prompt_length is None:
            raise ValueError(
                "`max_prompt_length` cannot be `None` when calling "
                "`dpo_preprocess`. Either pass a non-`None` value to "
                "`dpo_preprocess` or set the preprocessor attribute: "
                "`preprocessor.max_prompt_length = ...`."
            )

        # Tokenise and left-pad the prompts.
        prompts = x["prompts"]
        prompt_tokens = self.tokenizer(prompts)
        prompt_tokens, _ = self.packer(
            prompt_tokens,
            sequence_length=max_prompt_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
            padding_side="left",
        )

        # For computations below, let's get the batch size. It is easier to get
        # the batch size instead of the input, because the input can be a tensor
        # or a Python list, etc.
        batch_size = tf.shape(prompt_tokens)[0]

        def _process(prompt_tokens, responses):
            response_tokens = self.tokenizer(responses)
            token_ids, _ = self.packer(
                (prompt_tokens, response_tokens),
                sequence_length=sequence_length,
                add_start_value=False,
                add_end_value=self.add_end_token,
            )
            return token_ids

        chosen_responses = x["chosen_responses"]
        rejected_responses = x["rejected_responses"]
        prompt_chosen_tokens = _process(prompt_tokens, chosen_responses)
        prompt_rejected_tokens = _process(prompt_tokens, rejected_responses)

        # `(batch_size, sequence_length)` -> `(2 * batch_size, sequence_length)`
        # Can do a model forward pass on all both chosen and rejected samples in
        # one go.
        prompt_response_tokens = tf.concat(
            (prompt_chosen_tokens, prompt_rejected_tokens), axis=0
        )

        # Compute padding mask and response mask.
        padding_mask = prompt_response_tokens != self.tokenizer.pad_token_id
        response_mask = tf.concat(
            (
                tf.zeros((2 * batch_size, max_prompt_length)),
                tf.ones((2 * batch_size, sequence_length - max_prompt_length)),
            ),
            axis=1,
        )
        response_mask = tf.cast(response_mask, "bool")
        response_mask = tf.logical_and(response_mask, padding_mask)

        x = {
            "token_ids": prompt_response_tokens,
            "padding_mask": padding_mask,
            "response_mask": response_mask,
        }

        return keras.utils.pack_x_y_sample_weight(x, None, None)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert strings to integer token input for generation.

        Similar to calling the layer for training, this method takes in strings
        or tensor strings, tokenizes and packs the input, and computes a padding
        mask masking all inputs not filled in with a padded value.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        if not self.built:
            self.build(None)

        x = self.tokenizer(x)
        token_ids, padding_mask = self.packer(
            x, sequence_length=sequence_length, add_end_value=False
        )
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    @preprocessing_function
    def generate_postprocess(
        self,
        x,
    ):
        """Convert integer token output to strings for generation.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        if not self.built:
            self.build(None)

        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        ids_to_strip = self.tokenizer.special_token_ids
        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
            }
        )
        return config

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value
        if self.packer is not None:
            self.packer.sequence_length = value

    @property
    def max_prompt_length(self):
        return self._max_prompt_length

    @max_prompt_length.setter
    def max_prompt_length(self, value):
        self._max_prompt_length = value

    def export_to_transformers(self, path):
        """Export the preprocessor to HuggingFace Transformers format.

        Args:
            path: str. Path to save the exported preprocessor/tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError("Preprocessor must have a tokenizer for export.")
        from keras_hub.src.utils.transformers.export.hf_exporter import (
            export_tokenizer,
        )

        export_tokenizer(self.tokenizer, path)
