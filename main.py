from utils.data_loader_new import prepare_data_seq
from utils import config
from model.transformer import Transformer
from model.transformer_mulexpert import Transformer_experts
from model.common_layer import evaluate, count_parameters, make_infinite, NoamOpt, get_input_from_batch, get_output_from_batch
import tensorflow as tf
from tensorflow.keras import layers
from copy import deepcopy
from tqdm import tqdm
import os
import math
import time
import datetime
import numpy as np 

np.random.seed(0)

best_ppl = 1000
check_iter = 2000
data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)
data_iter = make_infinite(data_loader_tra)
save_model_path ='./saved_model/moel'

if(config.model == "trs"):
    model = Transformer(vocab,decoder_number=program_number)
elif(config.model == "experts"):
    model = Transformer_experts(vocab,decoder_number=program_number)
print("MODEL USED",config.model)
print("TRAINABLE PARAMETERS",count_parameters(model))

def train(model, batch, data_loader_val):
    patient = 0
    uniq_cfg_name = datetime.datetime.now().strftime("%Y")
    checkpoint_prefix = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_prefix):
        print("create model dir: %s" % checkpoint_prefix)
        os.mkdir(checkpoint_prefix)

    checkpoint_path = os.path.join(checkpoint_prefix, uniq_cfg_name)
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("load weight from: %s" % checkpoint_path)

    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    if (config.label_smoothing):
        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none', label_smoothing=0.1)
    optimizer = tf.keras.optimizers.Adam(lr=config.lr)
    if(config.noam):
        optimizer = NoamOpt(config.hidden_dim, 1, 8000, tf.keras.optimizers.Adam(lr=0, beta_1=0.9, beta_2=0.98, epsilon=1e-9))
    
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')

    @tf.function 
    def train_step(input_x, training = True):
        dec_batch, _, _, _, _ = get_output_from_batch(input_x)
        if training:
            with tf.GradientTape() as tape:
                logit, logit_prob = model(input_x, training)
                train_loss = criterion(tf.reshape(logit, [-1, logit.shape[-1]]), tf.reshape(dec_batch, -1)) + criterion(logit_prob, tf.cast(input_x['program_label'], dtype=tf.int64))
            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_pred_program = np.argmax(logit_prob.numpy(), axis=1)
        else:
            logit, logit_prob = model(input_x, training)
            test_loss = criterion(tf.reshape(logit, [-1, logit.shape[-1]]), tf.reshape(dec_batch, -1)) + criterion(logit_prob, tf.cast(input_x['program_label'], dtype=tf.int64))
            test_pred_program = np.argmax(logit_prob.numpy(), axis=1)
        train_loss.update_state(train_loss)
        train_accuracy.update_state(input_x["program_label"], train_pred_program)
        test_loss.update_state(test_loss)
        test_accuracy.update_state(input_x["program_label"], test_pred_program)

    try:
        train_loss.reset_states()
        train_accuracy.reset_states()
        for n_iter in tqdm(range(1000000)):
            train_step(batch, training = True)        
            if((n_iter+1)%check_iter==0):
                print("[train iter %d] [%s]: %0.3f [%s]: %0.3f  [%s]: %0.3f" %  (n_iter+1, "loss", train_loss.result(), "ppl", math.exp(train_loss.result()), "emo_acc", train_accuracy.result()))
                train_loss.reset_states()
                train_accuracy.reset_states()

                model.save_weights(checkpoint_path, overwrite=True)

                test_loss.reset_states()
                test_accuracy.reset_states()
                pbar = tqdm(enumerate(data_loader_val),total=len(data_loader_val))
                for j, test_batch in pbar:
                    train_step(test_batch, training = False)
                print("[test iter %d] [%s]: %0.3f [%s]: %0.3f  [%s]: %0.3f" %  (n_iter+1, "loss", test_loss.result(), "ppl", math.exp(test_loss.result()), "emo_acc", test_accuracy.result()))
                
                if (config.model == "experts" and n_iter<13000):
                    continue
                if math.exp(test_loss.result()) < best_ppl:
                    model.save(save_model_path)
                    best_ppl = math.exp(test_loss.result())
                else:
                    patient += 1
                if patient > 2:
                    break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if not config.test:
    train(model, next(data_iter), data_loader_val)
else:
    test_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')
    
    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    if (config.label_smoothing):
        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none', label_smoothing=0.1)
    
    model = tf.keras.models.load_model(save_model_path)
    pbar = tqdm(enumerate(data_loader_val),total=len(data_loader_val))
    for j, test_batch in pbar:
        dec_batch, _, _, _, _ = get_output_from_batch(test_batch)
        logit, logit_prob = model(test_batch, training=False)
        test_loss = criterion(tf.reshape(logit, [-1, logit.shape[-1]]), tf.reshape(dec_batch, -1)) + criterion(logit_prob, tf.cast(test_batch['program_label'], dtype=tf.int64))
        test_pred_program = np.argmax(logit_prob.numpy(), axis=1)
        test_loss.update_state(test_loss)
        test_accuracy.update_state(test_batch["program_label"], test_pred_program)
    print("[test metrics] [%s]: %0.3f [%s]: %0.3f  [%s]: %0.3f" %  ("loss", test_loss.result(), "ppl", math.exp(test_loss.result()), "emo_acc", test_accuracy.result()))

