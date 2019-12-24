import tensorflow as tf
from tensorflow import keras
from nlp.bert4tf2 import tf_utils
from nlp.bert4tf2 import bert_modeling
from nlp.bert4tf2 import networks


class BertModels(object):

    def __init__(self):
        pass

    def get_transformer_encoder(self, bert_config, sequence_length, float_dtype=tf.float32):
        """Gets a 'TransformerEncoder' object.
        :args
            bert_config: Model configs about bert or other variant
            sequence_length: Maximum sequence length of the training data.
            float_dtype: tf.dtype, tf.float32 or tf.float16.
        :returns
            A networks. TransformerEncoder object.
        """

        kwargs = dict(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            num_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            activation=tf_utils.get_activation(bert_config.hidden_act),
            dropout_rate=bert_config.hidden_dropout_prob,
            attention_dropout_rate=bert_config.attention_probs_dropout_prob,
            sequence_length=sequence_length,
            max_sequence_length=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range),
            float_dtype=float_dtype.name)

        if isinstance(bert_config, bert_modeling.AlbertConfig):
            kwargs['embedding_width'] = bert_config.embedding_size
            return networks.AlbertTransformerEncoder(**kwargs)
        else:
            assert isinstance(bert_config, bert_modeling.BertConfig)
            return networks.TransformerEncoder(**kwargs)

    def classifier_model(self, bert_config, float_type, num_labels, max_seq_length, final_layer_initializer=None,
                         hub_module_url=None):
        """Bert classifier model

        Construct a Keras model for predicting `num_labels` outputs from an input with
        maximum sequence length `max_seq_length`.

        :arg
            bert_config: BertConfig or AlbertConfig, the config defines the core BERT or ALBERT model.
            float_type: dtype, tf.float32 or tf.bfloat16.
            num_labels: integer, the number of classes.
            max_seq_length: integer, the maximum input sequence length.
            final_layer_initializer: Initializer for final dense layer,Defaulted TruncatedNormal initializer.
            hub_module_url: TF-Hub path/url to Bert module.
        :returns
            Combined prediction model (words, mask, type) -> (one-hot labels)
            Bert sub-model (words, mask, type) -> (bert_outputs)
        """
        if final_layer_initializer is not None:
            initializer = final_layer_initializer
        else:
            initializer = tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range)

        if not hub_module_url:
            bert_encoder = self.get_transformer_encoder(bert_config, max_seq_length)
            return networks.bert_classifier.BertClassifier(
                bert_encoder,
                num_classes=num_labels,
                dropout_rate=bert_config.hidden_dropout_prob,
                initializer=initializer), bert_encoder
