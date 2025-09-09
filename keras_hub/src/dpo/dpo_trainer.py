import keras
from keras import ops

from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.pipeline_model import _convert_inputs_to_dataset
from keras_hub.src.utils.pipeline_model import _train_validation_split

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _large_negative_number(dtype):
    """Return a large negative number based on dtype."""
    if keras.backend.standardize_dtype(dtype) == "float16":
        return -3e4
    return -1e9


def _log_softmax(x, mask=None, axis=-1):
    if mask is not None:
        mask = ops.expand_dims(mask, axis=-1)
        negative_mask = ops.logical_not(mask)
        cast_mask = ops.multiply(
            ops.cast(negative_mask, x.dtype), _large_negative_number(x.dtype)
        )
        x = ops.where(mask, x, cast_mask)

    return ops.log_softmax(x, axis=axis)


def _selective_log_softmax(logits, token_ids, mask):
    log_probs = _log_softmax(logits, mask=mask, axis=-1)
    per_token_log_probs = ops.take_along_axis(
        log_probs, token_ids[..., None], axis=-1
    )
    return per_token_log_probs[..., 0]


def _compute_log_probs(logits, token_ids, padding_mask, response_start_idx):
    # Suppose logits are L0 to L10, and tokens are T0 to T10. The prediction
    # for T1 comes from L0, T2 L1, etc.
    logits = logits[:, response_start_idx - 1 : -1, :]
    token_ids = token_ids[:, response_start_idx:]
    padding_mask = padding_mask[:, response_start_idx:]

    per_token_log_probs = _selective_log_softmax(
        logits, token_ids, mask=padding_mask
    )
    per_token_log_probs = ops.multiply(per_token_log_probs, padding_mask)

    chosen_log_probs, rejected_log_probs = ops.split(
        per_token_log_probs, indices_or_sections=2, axis=0
    )
    return chosen_log_probs, rejected_log_probs


class DPOTrainer(keras.Model):
    def __init__(
        self,
        model,
        reference_model,
        preprocessor=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        # Models
        self.model = model
        self.reference_model = reference_model

        # Freeze the reference model.
        self.reference_model.trainable = False

        # Preprocessor
        self.preprocessor = (
            model.preprocessor if model.preprocessor else preprocessor
        )

        # Save a private copy of the input format.
        self._input_keys = tuple(self.model.input.keys())

    def build(self, input_shape):
        self.model.build(input_shape)
        self.reference_model.build(input_shape)

    def call(self, inputs):
        logits = self.model(self._get_inputs(inputs))
        return logits

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="top_k",
        beta=1.0,
        **kwargs,
    ):
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(2e-5)
        if loss == "auto":
            loss = keras.losses.BinaryCrossentropy(
                from_logits=True, label_smoothing=0.1
            )
        if weighted_metrics == "auto":
            weighted_metrics = [keras.metrics.SparseCategoricalAccuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            **kwargs,
        )

        self.beta = beta

    def preprocess_samples(self, x, y=None, sample_weight=None):
        # If `preprocessor` is `None`, return inputs unaltered.
        if self.preprocessor is None:
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        # If `preprocessor` is `Preprocessor` subclass, pass labels as a kwarg.
        if isinstance(self.preprocessor, Preprocessor):
            return self.preprocessor.dpo_preprocess(
                x, y=y, sample_weight=sample_weight
            )
        # For other layers and callable, do not pass the label.
        x = self.preprocessor.dpo_preprocess(x)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        **kwargs,
    ):
        if validation_split and validation_data is None:
            (x, y, sample_weight), validation_data = _train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        x = _convert_inputs_to_dataset(x, y, sample_weight, batch_size)
        x = x.map(
            self.preprocess_samples, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        if validation_data is not None:
            if not isinstance(validation_data, tf.data.Dataset):
                (vx, vy, vsw) = keras.utils.unpack_x_y_sample_weight(
                    validation_data
                )
                validation_data = _convert_inputs_to_dataset(
                    vx, vy, vsw, batch_size
                )

        return super().fit(
            x=x,
            y=None,
            batch_size=None,
            sample_weight=None,
            validation_data=validation_data,
            **kwargs,
        )

    def predict(**kwargs):
        raise NotImplementedError(
            "`predict` is not implemented for DPOTrainer. Please call "
            "`predict` separately on `self.model`."
        )

    def evaluate(**kwargs):
        raise NotImplementedError(
            "`evaluate` is not implemented for DPOTrainer. Please call "
            "`evaluate` separately on `self.model`."
        )

    def _get_inputs(self, inputs):
        return {key: inputs[key] for key in self._input_keys}

    def _get_reference_log_probs(self, inputs):
        # Unpack the input dictionary.
        token_ids, padding_mask = (inputs["token_ids"], inputs["padding_mask"])

        reference_logits = self.reference_model(self._get_inputs(inputs))
        reference_chosen_log_probs, reference_rejected_log_probs = (
            _compute_log_probs(
                reference_logits,
                token_ids=token_ids,
                padding_mask=padding_mask,
                response_start_idx=self.preprocessor.max_prompt_length,
            )
        )

        return reference_chosen_log_probs, reference_rejected_log_probs

    def compute_loss(self, x, y, y_pred, sample_weight=None, training=True):
        # === Unpack inputs ===
        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        logits = y_pred

        # === Compute reference log probs ===
        reference_chosen_log_probs, reference_rejected_log_probs = (
            self._get_reference_log_probs(x)
        )
        # Ensure we don't back prop on this.
        reference_chosen_log_probs = ops.stop_gradient(
            reference_chosen_log_probs
        )
        reference_rejected_log_probs = ops.stop_gradient(
            reference_rejected_log_probs
        )

        # === Compute log probs using the policy model ===
        chosen_log_probs, rejected_log_probs = _compute_log_probs(
            logits,
            token_ids=token_ids,
            padding_mask=padding_mask,
            response_start_idx=self.preprocessor.max_prompt_length,
        )

        # === Compute loss ===
        chosen_rewards = chosen_log_probs - reference_chosen_log_probs
        rejected_rewards = rejected_log_probs - reference_rejected_log_probs
        margin = chosen_rewards - rejected_rewards
        beta_margin = self.beta * margin

        # TODO: Consider moving this to the preprocessor.
        y = ops.ones_like(margin)
        loss = self.loss(y_true=y, y_pred=beta_margin)

        return loss
