"""
LSTM model architecture for demand forecasting
"""

import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import Optional, List, Tuple

from .config import ForecastingConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """Attention mechanism for LSTM"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        # Compute attention scores
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Weighted sum
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self):
        return super(AttentionLayer, self).get_config()


class LSTMForecaster:
    """LSTM model for demand forecasting"""

    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        self.model = None
        self.history = None

    def build_model(
        self,
        input_shape: Tuple[int, int],
        lstm_units: Optional[List[int]] = None,
        dense_units: Optional[List[int]] = None,
        dropout_rate: Optional[float] = None,
        use_attention: Optional[bool] = None
    ) -> Model:
        """
        Build LSTM model architecture

        Args:
            input_shape: Shape of input (sequence_length, n_features)
            lstm_units: List of LSTM layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanism

        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model")
        logger.info(f"Input shape: {input_shape}")

        lstm_units = lstm_units or self.config.LSTM_UNITS
        dense_units = dense_units or self.config.DENSE_UNITS
        dropout_rate = dropout_rate or self.config.DROPOUT_RATE
        use_attention = use_attention if use_attention is not None else self.config.USE_ATTENTION

        # Input layer
        inputs = layers.Input(shape=input_shape, name='input')

        # LSTM layers
        x = inputs
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1) or use_attention

            if self.config.USE_BIDIRECTIONAL:
                x = layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=dropout_rate,
                        recurrent_dropout=self.config.RECURRENT_DROPOUT_RATE,
                        name=f'bidirectional_lstm_{i+1}'
                    )
                )(x)
            else:
                x = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=self.config.RECURRENT_DROPOUT_RATE,
                    name=f'lstm_{i+1}'
                )(x)

            logger.info(f"Added LSTM layer {i+1}: {units} units, return_sequences={return_sequences}")

        # Attention layer
        if use_attention:
            x = AttentionLayer(name='attention')(x)
            logger.info("Added attention layer")

        # Dense layers
        for i, units in enumerate(dense_units):
            x = layers.Dense(
                units,
                activation=self.config.ACTIVATION,
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
            logger.info(f"Added dense layer {i+1}: {units} units")

        # Output layer
        outputs = layers.Dense(
            1,
            activation=self.config.OUTPUT_ACTIVATION,
            name='output'
        )(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='LSTM_Forecaster')

        # Compile model
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss=self.config.LOSS,
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )

        logger.info(f"Model compiled with optimizer={self.config.OPTIMIZER}, loss={self.config.LOSS}")
        logger.info(f"Total parameters: {model.count_params():,}")

        self.model = model
        return model

    def _get_optimizer(self):
        """Get optimizer based on config"""
        lr = self.config.LEARNING_RATE

        if self.config.OPTIMIZER.lower() == 'adam':
            return Adam(learning_rate=lr)
        elif self.config.OPTIMIZER.lower() == 'rmsprop':
            return RMSprop(learning_rate=lr)
        elif self.config.OPTIMIZER.lower() == 'sgd':
            return SGD(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")

    def summary(self):
        """Print model summary"""
        if self.model is None:
            logger.warning("Model not built yet")
            return

        self.model.summary()

    def get_model(self) -> Optional[Model]:
        """Get the built model"""
        return self.model

    def save_model(self, path: Optional[str] = None):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not built yet")

        path = path or self.config.get_model_path()
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Optional[str] = None):
        """Load model from disk"""
        path = path or self.config.get_model_path()

        # Register custom layers
        custom_objects = {'AttentionLayer': AttentionLayer}

        self.model = keras.models.load_model(path, custom_objects=custom_objects)
        logger.info(f"Model loaded from {path}")

        return self.model

    def get_model_info(self) -> dict:
        """Get model information"""
        if self.model is None:
            return {'status': 'not_built'}

        return {
            'status': 'built',
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in self.model.trainable_weights]),
            'layers': len(self.model.layers),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'optimizer': self.config.OPTIMIZER,
            'loss': self.config.LOSS
        }
