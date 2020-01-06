# -*- coding: utf-8 -*-

"""
deep Neural Networks obtain categorcial features embedding
"""
import os
import pandas as pd
import numpy as np
import random as rd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from ..metrics import ks

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam, Nadam


class EarlyStopping(Callback):

    def __init__(self, training_data=False, validation_data=False, testing_data=False, min_delta=0, patience=0,
                 test_check=False, model_file=None, scoring=ks, verbose=0):
        super(EarlyStopping, self).__init__()
        self.best_epoch = 0
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.test_check = test_check
        self.scoring = scoring

        if training_data:
            self.x_tr = training_data[0]
            self.y_tr = training_data[1]
        else:
            self.x_tr = False
            self.y_tr = False
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        if testing_data:
            self.x_te = testing_data[0]
            self.y_te = testing_data[1]
        else:
            self.x_te = False
            self.y_te = False
        self.model_file = model_file

    def on_train_begin(self, logs={}):
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ', self.best_epoch, ': EarlyStopping')

    def on_epoch_end(self, epoch, logs={}):
        if self.x_tr:
            y_pred = self.model.predict(self.x_tr, batch_size=100000)
            score_tr = self.scoring(self.y_tr, y_pred)
        else:
            score_tr = 0

        y_hat_val = self.model.predict(self.x_val, batch_size=100000)
        score_val = self.scoring(self.y_val, y_hat_val)

        if self.x_te:
            y_hat_te = self.model.predict(self.x_te, batch_size=100000)
            score_te = self.scoring(self.y_te, y_hat_te)
        else:
            score_te = 0
        metric_name = getattr(self.scoring, '__name__')
        print('{0}_train: {1} - {0}_val: {2} - {0}_test: {3}'.format(metric_name, str(round(score_tr, 6)),
                                                                     str(round(score_val, 6)),
                                                                     str(round(score_te, 6))), end=100*' '+'\n')

        if self.model_file:
            print("saving", self.model_file+'.'+str(epoch))
            self.model.save_weights(self.model_file+'.'+str(epoch))

        if self.x_val:
            if self.test_check:
                current = score_te
            else:
                current = score_val
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.best_epoch = epoch
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True


class DNN(BaseEstimator, ClassifierMixin):

    def __init__(self, train, valid, test, predictors, categorical_features, target, params, fixemb=True,
                 sameNDenseAsEmb=True, numeDropout=True, NumeBatchNormalization=True, train_check=True,
                 test_check=False, seed=1024):
        self.train = train
        self.valid = valid
        self.test = test
        self.predictors = predictors
        self.categorical_features = categorical_features
        self.target = target
        self.params = params
        self.fixemb = fixemb
        self.sameNDenseAsEmb = sameNDenseAsEmb
        self.numeDropout = numeDropout
        self.NumeBatchNormalization = NumeBatchNormalization
        self.train_check = train_check
        self.test_check = test_check
        self.seed = seed

    def fit(self, X):
        np.random.seed(self.seed)
        rd.seed(self.seed)
        trn_x = self.train[self.predictors]
        vld_x = self.valid[self.predictors]
        test_x = self.test[self.predictors]
        trn_y = self.train[self.target]
        vld_y = self.valid[self.target]
        test_y = self.test[self.target]
        batch_size = int(self.params['batch_size'])
        epochs_for_lr = float(self.params['epochs_for_lr'])
        max_epochs = int(self.params['max_epochs'])
        emb_cate = int(self.params['emb_cate'])
        dense_cate = int(self.params['dense_cate'])
        dense_nume_n_layers = int(self.params['dense_nume_n_layers'])
        drop = float(self.params['drop'])
        lr = float(self.params['lr'])
        lr_init = float(self.params['lr_init'])
        lr_fin = float(self.params['lr_fin'])
        n_layers = int(self.params['n_layers'])
        patience = int(self.params['patience'])
        optim = self.params['optimizer']
        lastdropout = self.params['lastdropout']
        batchnormalization = self.params['batchnormalization']

        train_dict = {}
        valid_dict = {}
        test_dict = {}
        input_list = []
        emb_list = []
        numerical_feats = []
        tot_emb_n = 0
        for col in self.predictors:
            if col not in self.categorical_features:
                numerical_feats.append(col)

        if len(self.categorical_features) > 0:
            for col in self.categorical_features:
                train_dict[col] = np.array(trn_x[col])
                valid_dict[col] = np.array(vld_x[col])
                test_dict[col] = np.array(test_x[col])
                inpt = Input(shape=[1], name=col)
                input_list.append(inpt)
                max_val = np.max([trn_x[col].max(), vld_x[col].max(), test_x[col].max()]) + 1
                emb_n = np.min([emb_cate, max_val])
                if self.fixemb:
                    emb_n = emb_cate
                tot_emb_n += emb_n
                if emb_n == 1:
                    print("emb_1 = 1")
                    return 0
                print('Embedding size:', max_val, emb_cate, trn_x[col].max(), vld_x[col].max(), test_x[col].max(), emb_n,
                      col)
                embd = Embedding(max_val, emb_n)(inpt)
                emb_list.append(embd)
            if len(emb_list) == 1:
                print("emb_list = 1")
                return 0
            fe = concatenate(emb_list)
            s_dout = SpatialDropout1D(drop)(fe)
            x1 = Flatten()(s_dout)

        if self.sameNDenseAsEmb:
            dense_cate = tot_emb_n
        if len(numerical_feats) > 0:
            train_dict['numerical'] = trn_x[numerical_feats].values
            valid_dict['numerical'] = vld_x[numerical_feats].values
            test_dict['numerical'] = test_x[numerical_feats].values
            inpt = Input((len(numerical_feats),), name='numerical')
            input_list.append(inpt)
            x2 = inpt
            for n in range(dense_nume_n_layers):
                x2 = Dense(dense_cate, activation='relu', kernel_initializer=RandomUniform(seed=self.seed))(x2)
                if self.numeDropout:
                    x2 = Dropout(drop)(x2)
                if self.NumeBatchNormalization:
                    x2 = BatchNormalization()(x2)

        if len(numerical_feats) > 0 and len(self.categorical_features) > 0:
            x = concatenate([x1, x2])
        elif len(numerical_feats) > 0:
            x = x2
        elif len(self.categorical_features) > 0:
            x = x1
        else:
            return 0  # for small data test

        for n in range(n_layers):
            x = Dense(dense_cate, activation='relu', kernel_initializer=RandomUniform(seed=self.seed))(x)
            if lastdropout:
                x = Dropout(drop)(x)
            if batchnormalization:
                x = BatchNormalization()(x)
        outp = Dense(1, activation='sigmoid', kernel_initializer=RandomUniform(seed=self.seed))(x)
        model = Model(inputs=input_list, outputs=outp)
        if optim == 'adam':
            optimizer = Adam(lr=lr)
        elif optim == 'nadam':
            optimizer = Nadam(lr=lr)
        else:
            def exp_decay(init, fin, _steps):
                return (init / fin) ** (1 / (_steps - 1)) - 1
            steps = int(len(trn_x) / batch_size) * epochs_for_lr
            lr_decay = exp_decay(lr_init, lr_fin, steps)
            optimizer = Adam(lr=lr, decay=lr_decay)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        model.summary()

        model_file = './weight-' + str(os.getpid()) + '.hdf5'
        if self.train_check:
            training_data = (train_dict, trn_y)
        else:
            training_data = False
        if self.test_check:
            testing_data = (test_dict, test_y)
        else:
            testing_data = False

        earlystopping = EarlyStopping(
            training_data=training_data,
            validation_data=(valid_dict, vld_y),
            testing_data=testing_data,
            patience=patience,
            model_file=model_file,
            test_check=False,
            verbose=1)

        class_weight = {0: .01, 1: .99}
        model.fit(train_dict,
                  trn_y,
                  validation_data=[valid_dict, vld_y],
                  batch_size=batch_size,
                  class_weight=class_weight,
                  epochs=max_epochs,
                  shuffle=True,
                  verbose=2,
                  callbacks=[earlystopping])

        best_epoch = earlystopping.best_epoch
        print('loading', model_file + '.' + str(best_epoch))
        model.load_weights(model_file + '.' + str(best_epoch))
        os.system('rm -f ' + model_file + '.*')
        self.model = model
        self.earlystopping = earlystopping
        self.valid_dict = valid_dict
        self.test_dict = test_dict
        return self

    def predict(self, X=None):
        valid_preds = pd.DataFrame([i+1 for i in range(self.valid.shape[0])], columns=['index'])
        test_preds = self.model.predict(self.test_dict, batch_size=int(self.params['batch_size']),
                                        verbose=2)[:, 0]
        valid_preds['pred'] = self.model.predict(self.valid_dict, batch_size=int(self.params['batch_size']),
                                                 verbose=2)[:, 0]

        metric_name = getattr(self.earlystopping.scoring, '__name__')
        score = self.earlystopping.scoring(self.valid[self.target], valid_preds['pred'])
        print("valid dataset {}. {}".format(metric_name, score))
        return test_preds

    def predict_proba(self, X=None):
        valid_preds = pd.DataFrame([i+1 for i in range(self.valid.shape[0])], columns=['index'])
        test_preds = self.model.predict(self.test_dict, batch_size=int(self.params['batch_size']),
                                        verbose=2)
        valid_preds['pred'] = self.model.predict(self.valid_dict, batch_size=int(self.params['batch_size']),
                                                 verbose=2)[:, 0]

        metric_name = getattr(self.earlystopping.scoring, '__name__')
        score = self.earlystopping.scoring(self.valid[self.target], valid_preds['pred'])
        print("valid dataset {}. {}".format(metric_name, score))
        return test_preds
