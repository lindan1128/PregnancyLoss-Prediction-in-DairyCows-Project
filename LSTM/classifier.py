__author__ = "Dan LIN"

import os

from tensorflow import keras
from tensorflow.keras.layers import Layer
from keras import backend as K
from keras.callbacks import TensorBoard

from utils import specificity, f1_metric

class T2V(Layer):
    """ Time2Vec
    Adapted from the https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    """ 
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                shape=(1, self.output_dim),
                                initializer='uniform',
                                trainable=True)
        self.P = self.add_weight(name='P',
                                shape=(1, self.output_dim),
                                initializer='uniform',
                                trainable=True)
        self.w = self.add_weight(name='w',
                                shape=(1, 1),
                                initializer='uniform',
                                trainable=True)
        self.p = self.add_weight(name='p',
                                shape=(1, 1),
                                initializer='uniform',
                                trainable=True)
        super(T2V, self).build(input_shape)
        
    def call(self, x):
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        return K.concatenate([sin_trans, original], -1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'output_dim': self.output_dim})
        return config

class Attention(Layer):
    """ Attention layer
    Adapted from the https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/
    """ 
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)
    
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return alpha, context

class Att_LSTMNetwork():
    """ Attention-based Long Short-Term Memory (Att-LSTM)
    """
    def __init__(self):
        self.random_state = None

    def build_network(self, **kwargs):
        """
        Construct a network and return its input and output layers
        Returns
        -------
        input_layer : a keras layer
        meta_input_layer : a keras layer
        output_layer : a keras layer
        """
        input_layer = keras.layers.Input(shape=(14,1))
        meta_input_layer = keras.layers.Input(shape=(2,))
        time2vec = T2V(64)(input_layer)
        lstm_output_layer = keras.layers.LSTM(
            units=20,
            activation='relu',
            return_sequences=True)(time2vec)  
        att_scores, att_layer = Attention()(lstm_output_layer)
        output_layer = K.concatenate([att_layer, meta_input_layer]) 

        return input_layer, meta_input_layer, output_layer

class Att_LSTMClassifier(Att_LSTMNetwork):
    """ Attention-based Long Short-Term Memory (Att-LSTM)
    """

    def __init__(
            self,
            nb_epochs=100,
            batch_size=32,
            callbacks=None,
            random_state=0,
            verbose=False
    ):
        """
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param random_state: int, seed to any needed random actions
        """
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose

        self._is_fitted = False

    def build_model(self, nb_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model
        Returns
        -------
        output : a compiled Keras Model
        """
        input_layer, meta_input_layer, output_layer = self.build_network(**kwargs)
        output_layer = keras.layers.Dense(64, activation='relu')(output_layer)
        output_layer = keras.layers.Dropout(0.1)(output_layer)
        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(output_layer)
        model = keras.models.Model(inputs=[input_layer, meta_input_layer], outputs=output_layer)

        model.compile(
            loss = "categorical_crossentropy",
            optimizer = keras.optimizers.Adam(),
            metrics = ['accuracy', 'Recall', specificity, f1_metric],
        )

        if self.callbacks is None:
            self.callbacks = []

        self.callbacks = TensorBoard(log_dir='./save/lstm_logs',
                 histogram_freq=1,
                 batch_size=32,
                 write_graph=True,
                 write_grads=True,
                 write_images=True,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

        return model

    def fit(self, X_train, meta_train, y_train, X_val, meta_val, y_val, X_test, meta_test, y_test, **kwargs):
        """
        Fit the Classifier on the training set ([X, meta], y)
        ----------
        X_train/val/test : a pd.Dataframe, or a array-like of
        shape = (n_instances, series_length, n_dimensions)
        meta_train/test: a pd.Dataframe, or a array-like of
        shape = (n_instances, n_dimensions)
        y_train/val/test : a pd.Dataframe, or a array-like of
        shape = (n_instances, n_class)
            The training data class labels.
        Returns
        -------
        self : object
        """
        X_train = X_train
        meta_train = meta_train
        y_train = y_train
        X_val = X_val
        meta_val = meta_val
        y_val = y_val
        X_test = X_test
        meta_test = meta_test
        y_test = y_test

        self.batch_size = self.batch_size
        self.nb_classes = 2

        self.model = self.build_model(self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            [X_train, meta_train],
            y_train,
            validation_data=([X_val, meta_val], y_val),
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks
        )

        score, train_acc, train_sen, train_spec, train_f1 = self.model.evaluate(
            [X_train, meta_train],
            y_train,
            batch_size=self.batch_size)
        print('train_acc:', train_acc)
        print('train_sensitivity:', train_sen)
        print('train_specificity:', train_spec)
        print('train_F1:', train_f1)

        score, test_acc, test_sen, test_spec, test_f1 = self.model.evaluate(
            [X_test, meta_test],
            y_test,
            batch_size=self.batch_size)
        print('test_acc:', test_acc)
        print('test_sensitivity:', test_sen)
        print('test_specificity:', test_spec)
        print('test_F1:', test_f1)

        self.model.save('./save/attlstm.h5')
        print('Model Saved!')

        return self



    