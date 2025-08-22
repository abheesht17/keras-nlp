import keras
from keras import ops

from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.pipeline_model import PipelineModel


def _large_negative_number(dtype):
    """Return a Large negative number based on dtype."""
    if keras.backend.standardize_dtype(dtype) == "float16":
        return -3e4
    return -1e9


def _log_softmax(x, mask=None, axis=-1):
    if mask is not None:
        cast_mask = ops.multiply(
            ops.cast(mask, x.dtype), _large_negative_number(x.dtype)
        )
        x = ops.where(mask, x, cast_mask)

    return ops.log_softmax(x, axis=axis)


def _selective_log_softmax(token_ids, logits, mask):
    log_probs = _log_softmax(logits, mask=mask, axis=-1)
    per_token_log_probs = ops.take_along_axis(
        log_probs, token_ids[..., None], axis=-1
    )
    return per_token_log_probs[..., 0]


def _compute_log_probs(logits, token_ids, response_mask):
    # Suppose logits are L0 to L10, and tokens are T0 to T10. The prediction
    # for T1 comes from L0, T2 L1, etc.
    logits = logits[:, :-1, :]
    token_ids = token_ids[:, 1:]
    response_mask = response_mask[:, 1:]

    per_token_log_probs = _selective_log_softmax(
        token_ids, logits, mask=response_mask
    )

    chosen_log_probs, rejected_log_probs = ops.split(
        per_token_log_probs, indices_or_sections=2, axis=0
    )
    return chosen_log_probs, rejected_log_probs


class DPOTrainer(PipelineModel):
    def __init__(
        self,
        model,
        reference_model,
        preprocessor=None,
    ):
        # Models
        self.model = model
        self.reference_model = reference_model

        self.preprocessor = (
            model.preprocessor if model.preprocessor else preprocessor
        )

    def call(self, inputs):
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]

        logits = self.model(token_ids, padding_mask=padding_mask)
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

    def predict(**kwargs):
        pass

    def evaluate(**kwargs):
        pass

    def _get_reference_log_probs(self, inputs):
        # Unpack the input dictionary.
        token_ids, padding_mask, response_mask = (
            inputs["token_ids"],
            inputs["padding_mask"],
            inputs["response_mask"],
        )

        reference_logits = self.reference_model(
            token_ids, padding_mask=padding_mask
        )
        reference_chosen_log_probs, reference_rejected_log_probs = (
            _compute_log_probs(
                reference_logits,
                token_ids=token_ids,
                response_mask=response_mask,
            )
        )

        return reference_chosen_log_probs, reference_rejected_log_probs

    def compute_loss(self, x, y, y_pred, sample_weight=None, training=True):
        # === Unpack inputs ===
        token_ids, response_mask = (
            x["token_ids"],
            x["response_mask"],
        )
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
            logits, token_ids=token_ids, response_mask=response_mask
        )

        # === Compute loss ===
        chosen_rewards = chosen_log_probs - reference_chosen_log_probs
        rejected_rewards = rejected_log_probs - reference_rejected_log_probs
        margin = chosen_rewards - rejected_rewards
        beta_margin = self.beta * margin

        # TODO: Consider moving this to the preprocessor.
        y = ops.ones_like(margin)

        loss = self.loss_fn(
            y=y, y_pred=beta_margin, sample_weight=response_mask
        )

        return loss
