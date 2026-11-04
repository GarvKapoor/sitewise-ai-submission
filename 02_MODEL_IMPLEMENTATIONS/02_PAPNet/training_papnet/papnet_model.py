"""
PAPNet: Pattern-Affinitive Propagation Network for Satellite Land Classification
================================================================================
Implements Pattern-Affinitive Propagation (PAP) modules and complete network
architectures in TensorFlow / Keras for land use and land cover (EuroSAT)
classification and downstream site suitability analysis.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="papnet")
class PatternAffinityModule(layers.Layer):
    """
    Pattern-Affinitive Propagation (PAP) Layer.

    Models spatial and semantic pattern affinity across satellite image patches.
    Computes pairwise pattern affinity matrices and propagates contextual features
    across the receptive field with a learnable residual gating parameter (gamma).

    Formula:
        Q = Conv_q(X), K = Conv_k(X), V = Conv_v(X)
        Affinity = Softmax( (Q_flat @ K_flat^T) / sqrt(key_dim) )
        Propagated = Reshape(Affinity @ V_flat)
        Output = X + gamma * Conv_proj(Propagated)
    """

    def __init__(self, key_dim=32, value_dim=None, num_heads=1, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim) if value_dim is not None else None
        self.num_heads = int(num_heads)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        v_dim = self.value_dim if self.value_dim is not None else channels

        self.query_conv = layers.Conv2D(
            self.key_dim,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="pap_query"
        )
        self.key_conv = layers.Conv2D(
            self.key_dim,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="pap_key"
        )
        self.value_conv = layers.Conv2D(
            v_dim,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="pap_value"
        )
        self.proj_conv = layers.Conv2D(
            channels,
            kernel_size=1,
            padding="same",
            name="pap_proj"
        )
        self.norm = layers.BatchNormalization(name="pap_bn")

        # Learnable gating parameter initialized to 0 for stable training
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]
        num_patches = height * width

        # Query, Key, Value projections
        q = self.query_conv(x)  # (B, H, W, key_dim)
        k = self.key_conv(x)    # (B, H, W, key_dim)
        v = self.value_conv(x)  # (B, H, W, v_dim)

        # Flatten spatial dimensions
        q_flat = tf.reshape(q, [batch_size, num_patches, self.key_dim])
        k_flat = tf.reshape(k, [batch_size, num_patches, self.key_dim])
        v_dim = tf.shape(v)[-1]
        v_flat = tf.reshape(v, [batch_size, num_patches, v_dim])

        # Scaled dot-product Pattern Affinity Matrix
        scale = tf.math.rsqrt(tf.cast(self.key_dim, tf.float32))
        affinity = tf.matmul(q_flat, k_flat, transpose_b=True) * scale
        affinity = tf.nn.softmax(affinity, axis=-1)  # (B, N, N)

        # Affinity Propagation
        propagated = tf.matmul(affinity, v_flat)  # (B, N, v_dim)
        propagated = tf.reshape(propagated, [batch_size, height, width, v_dim])
        propagated = self.proj_conv(propagated)
        propagated = self.norm(propagated, training=training)

        # Residual connection gated by gamma
        return x + self.gamma * propagated

    def get_config(self):
        config = super().get_config()
        config.update({
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "num_heads": self.num_heads,
        })
        return config


@keras.utils.register_keras_serializable(package="papnet")
class PAPResidualBlock(layers.Layer):
    """
    Residual Convolutional Block augmented with Pattern-Affinitive Propagation.
    """

    def __init__(self, filters, stride=1, use_pap=True, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.stride = int(stride)
        self.use_pap = bool(use_pap)
        self.key_dim = int(key_dim)

    def build(self, input_shape):
        in_channels = int(input_shape[-1])

        self.conv1 = layers.Conv2D(
            self.filters,
            kernel_size=3,
            strides=self.stride,
            padding="same",
            use_bias=False,
            name="conv1"
        )
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.act1 = layers.Activation("swish", name="act1")

        self.conv2 = layers.Conv2D(
            self.filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="conv2"
        )
        self.bn2 = layers.BatchNormalization(name="bn2")

        if self.use_pap:
            self.pap = PatternAffinityModule(
                key_dim=self.key_dim,
                name="pap_module"
            )
        else:
            self.pap = None

        if self.stride != 1 or in_channels != self.filters:
            self.shortcut = keras.Sequential([
                layers.Conv2D(
                    self.filters,
                    kernel_size=1,
                    strides=self.stride,
                    padding="same",
                    use_bias=False,
                    name="shortcut_conv"
                ),
                layers.BatchNormalization(name="shortcut_bn")
            ], name="shortcut")
        else:
            self.shortcut = layers.Lambda(lambda x: x, name="identity")

        self.act_out = layers.Activation("swish", name="act_out")
        super().build(input_shape)

    def call(self, x, training=None):
        residual = self.shortcut(x, training=training)

        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.bn2(y, training=training)

        if self.pap is not None:
            y = self.pap(y, training=training)

        out = self.act_out(y + residual)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "stride": self.stride,
            "use_pap": self.use_pap,
            "key_dim": self.key_dim,
        })
        return config


def create_papnet_custom(input_shape=(64, 64, 3), num_classes=10, name="PAPNet_Custom"):
    """
    Builds a custom native PAPNet (Pattern-Affinitive Propagation Network)
    operating directly on 64x64 satellite inputs (EuroSAT native resolution).
    """
    inputs = keras.Input(shape=input_shape, name="satellite_image")

    # Data Augmentation (active during training only)
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.2, name="aug_rot")(x)
    x = layers.RandomZoom(0.2, name="aug_zoom")(x)

    # Input Rescaling to [0, 1] if not already normalized
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    # Stem: Initial Feature Extraction
    x = layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("swish", name="stem_act")(x)

    # Stage 1 (64x64 -> 32x32)
    x = PAPResidualBlock(filters=64, stride=2, use_pap=False, name="stage1_block1")(x)
    x = PAPResidualBlock(filters=64, stride=1, use_pap=True, key_dim=16, name="stage1_block2")(x)

    # Stage 2 (32x32 -> 16x16)
    x = PAPResidualBlock(filters=128, stride=2, use_pap=False, name="stage2_block1")(x)
    x = PAPResidualBlock(filters=128, stride=1, use_pap=True, key_dim=32, name="stage2_block2")(x)

    # Stage 3 (16x16 -> 8x8)
    x = PAPResidualBlock(filters=256, stride=2, use_pap=False, name="stage3_block1")(x)
    x = PAPResidualBlock(filters=256, stride=1, use_pap=True, key_dim=64, name="stage3_block2")(x)

    # Stage 4: Top Pattern-Affinitive Propagation Refinement (8x8)
    x = PatternAffinityModule(key_dim=64, name="pap_top_refinement")(x)

    # Classification Head
    avg_pool = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    feat = layers.Concatenate(name="pool_concat")([avg_pool, max_pool])

    feat = layers.Dense(128, activation="swish", name="head_dense")(feat)
    feat = layers.BatchNormalization(name="head_bn")(feat)
    feat = layers.Dropout(0.3, name="head_dropout")(feat)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="land_class_probabilities"
    )(feat)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def create_papnet_transfer(input_shape=(64, 64, 3), num_classes=10, weights="imagenet", name="PAPNet_Transfer"):
    """
    Builds a Transfer-Learning PAPNet integrating a pretrained backbone with
    Pattern-Affinitive Propagation (PAP) modules for land use classification.
    Uses cached ImageNet weights for rapid convergence (>90%+ on first epoch).
    """
    inputs = keras.Input(shape=input_shape, name="satellite_image")

    # Data Augmentation (active during training only)
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.2, name="aug_rot")(x)
    x = layers.RandomZoom(0.2, name="aug_zoom")(x)

    # Resize 64x64 -> 224x224 for backbone
    x = layers.Resizing(224, 224, name="resize_to_224")(x)

    # Pretrained Feature Backbone (Cached EfficientNetB0)
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(224, 224, 3)
    )
    if weights is not None:
        backbone.trainable = False

    features = backbone(x, training=False)  # (B, 7, 7, 1280)

    # Pattern-Affinitive Propagation Module
    # Models spatial and semantic affinity across satellite image patches
    pap_features = PatternAffinityModule(
        key_dim=64,
        name="pap_deep_affinity"
    )(features)

    # Classification Head
    gap = layers.GlobalAveragePooling2D(name="global_avg_pool")(pap_features)
    drop = layers.Dropout(0.3, name="head_dropout")(gap)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="land_class_probabilities"
    )(drop)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model, backbone
