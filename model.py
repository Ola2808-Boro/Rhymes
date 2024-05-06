import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten,Dropout
import keras
from keras.utils import save_img
import matplotlib.pyplot as plt
import os
import json

PATH='C:/Users/olkab/Desktop/Project Rhyme/Rhymes/algorithm_dataset_all.csv'



def read_data(path:str)->pd.DataFrame:
    """
    Reads the dataset from the specified path and returns it as a DataFrame.
    
    Args:
        path (str): The path to the dataset CSV file.
    
    Returns:
        pd.DataFrame: The DataFrame containing the dataset.
    """
    df = pd.read_csv(path)
    print(f'Data length {len(df["base_word"])}')
    return df


def processing_data(df:pd.DataFrame):
    """
    Processes the dataset by tokenizing words and converting characters to numerical sequences.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.
    
    Returns:
        tokenizer: The tokenizer object.
        x: The processed input data.
        y: The target data.
    """
    # Tokenize words
    polish_alphabet = 'aąbcćdeęfghijklłmnńoópqrsśtuvwyzźż'
    tokenizer = Tokenizer(char_level=True, filters=None, lower=True)
    tokenizer.fit_on_texts(polish_alphabet)

    # Convert characters to numerical sequences
    seq_word1 = tokenizer.texts_to_sequences(df['base_word'])
    seq_word2 = tokenizer.texts_to_sequences(df['rhyme_candidate'])

    max_sequence_length = 20
    pad_word1 = pad_sequences(seq_word1, maxlen=max_sequence_length)
    pad_word2 = pad_sequences(seq_word2, maxlen=max_sequence_length)

    x = np.concatenate((pad_word1, pad_word2), axis=-1)
    y = df['score'].values

    return tokenizer,x,y,max_sequence_length

def split_data(x,y):
    """
    Splits the dataset into training and validation sets.
    
    Args:
        x: The input data.
        y: The target data.
    
    Returns:
        x_train, x_val, y_train, y_val: The split training and validation data.
    """

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f'Train dataset {len(x_train)} validation dataset {len(x_val)}')
    return x_train, x_val, y_train, y_val

def build_models(tokenizer,max_sequence_length):
    """
    Builds and compiles multiple models for training.
    
    Args:
        tokenizer: The tokenizer object.
    
    Returns:
        model1, model2, model3: The compiled models.
    """
    embedding_dim = 8  # Adjust as needed
    vocab_size = len(tokenizer.word_index) + 1

    model1 = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2*max_sequence_length),
        Flatten(),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1, activation='sigmoid')  # Output layer for binary classification
    ])

    model2=Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2*max_sequence_length),
                Flatten(),
                Dense(464,activation='relu'),
                Dense(50,activation='relu'),
                Dropout(0.2),
                Dense(25,activation='relu'),
                Dropout(0.2),
                Dense(16,activation='relu'),
                Dropout(0.2),
                Dense(8,activation='relu'),
                Dropout(0.2),
                Dense(1,activation='sigmoid'),
    ])

    model3 = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2*max_sequence_length),
        LSTM(units=64),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1, activation='sigmoid')  # Output layer for binary classification
    ])

    return model1,model2,model3

def predict_function(model,word_1, word_2,tokenizer):
    """
    Predicts the score for two words using a trained model.
    
    Args:
        model: The trained model.
        word_1 (str): The first word.
        word_2 (str): The second word.
        tokenizer: The tokenizer object.
    
    Returns:
        result: The predicted score.
        raw_result: The raw prediction result.
    """

    seq_word1 = tokenizer.texts_to_sequences([word_1])
    seq_word2 = tokenizer.texts_to_sequences([word_2])

    max_sequence_length = 20
    pad_word1 = pad_sequences(seq_word1, maxlen=max_sequence_length)
    pad_word2 = pad_sequences(seq_word2, maxlen=max_sequence_length)

    x = np.concatenate((pad_word1, pad_word2), axis=-1)
    raw_result=model.predict(x)[0]
    result=np.round(raw_result)
    return result,raw_result

def training(model_name:str,model,epochs:int,optimizer,batch_size:int,optimizer_name:str,x_train,y_train,x_val,y_val):
    
    """
    Trains a model and saves its weights.
    
    Args:
        model_name (str): The name of the model.
        model: The model to train.
        epochs (int): The number of epochs for training.
        optimizer: The optimizer used for training.
        batch_size (int): The batch size for training.
        optimizer_name (str): The name of the optimizer.
        x_train: The input training data.
        y_train: The target training data.
    
    Returns:
        history: The training history.
    """

    checkpoint_filepath = f'models/{model_name}/{model_name}_{epochs}_{batch_size}_{optimizer_name}.weights.best.hdf5'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

    callback = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=10,verbose=1)
    callbacks=[model_checkpoint_callback,callback]
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error',keras.metrics.R2Score()])
    history=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),callbacks=callbacks)

    return history

def plot_save_charts(model_setting,history):
    """
    Plots and saves charts for the training history of a model.
    
    Args:
        model_setting (dict): The settings of the model.
        history: The training history of the model.
    """

    os.makedirs(name=f'models/{model_setting["model_name"]}/charts',exist_ok=True)
    params=[['mean_absolute_error','val_mean_absolute_error'],['r2_score','val_r2_score'],['loss','val_loss']]
    for param in params:
        plt.plot(history.history[param[0]])
        plt.plot(history.history[param[1]])
        plt.title(f'Model {param[0]}')
        plt.ylabel(param[0])
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'models/{model_setting["model_name"]}/charts/{model_setting["model_name"]}_{model_setting["optimizer_name"]}_{model_setting["batch_size"]}_{param[0]}.png')
        plt.clf()


def train(settings):

    """
    Trains multiple models with different settings.
    
    Args:
        settings (list): A list containing dictionaries of model settings.
    """
   
    df=read_data(path=PATH)
    tokenizer,x,y,max_sequence_length=processing_data(df=df)
    x_train, x_val, y_train, y_val=split_data(x=x,y=y)
    model1,model2,model3=build_models(tokenizer=tokenizer,max_sequence_length=max_sequence_length)
    settings=[{
    'model':model1,
    'model_name':'model1',
    'epochs':30,
    'optimizer':keras.optimizers.SGD(learning_rate=0.1),  
    'optimizer_name':'SGD',  
    'batch_size':32
    },
    {
    'model':model1,
    'model_name':'model1',
    'epochs':30,
    'optimizer':keras.optimizers.Adam(learning_rate=0.1),
    'optimizer_name':'Adam',  
    'batch_size':64

    },
    {
        'model':model1,
        'model_name':'model1',
        'epochs':30,
        'optimizer':keras.optimizers.SGD(learning_rate=0.1),
        'optimizer_name':'SGD',  
        'batch_size':64

    },
    {
        'model':model1,
        'model_name':'model1',
        'epochs':30,
        'optimizer':keras.optimizers.Adam(learning_rate=0.1),
        'optimizer_name':'Adam',  
        'batch_size':32

    },
    {
        'model':model2,
        'model_name':'model2',
        'epochs':30,
        'optimizer':keras.optimizers.SGD(learning_rate=0.1),
        'optimizer_name':'SGD',  
        'batch_size':32

        },
        {
        'model':model2,
        'model_name':'model2',
        'epochs':30,
        'optimizer':keras.optimizers.Adam(learning_rate=0.1),
        'optimizer_name':'Adam',  
        'batch_size':64

    },
    {
        'model':model2,
        'model_name':'model2',
        'epochs':30,
        'optimizer':keras.optimizers.SGD(learning_rate=0.1),
        'optimizer_name':'SGD',  
        'batch_size':64

    },
    {
        'model':model2,
        'model_name':'model2',
        'epochs':30,
        'optimizer':keras.optimizers.Adam(learning_rate=0.1),
        'optimizer_name':'Adam',  
        'batch_size':32

    },
    {
        'model':model3,
        'model_name':'model3',
        'epochs':30,
        'optimizer':keras.optimizers.Adam(learning_rate=0.1),
        'optimizer_name':'Adam',  
        'batch_size':32

        },
        {
        'model':model3,
        'model_name':'model3',
        'epochs':30,
        'optimizer':keras.optimizers.SGD(learning_rate=0.1),
        'optimizer_name':'SGD',  
        'batch_size':64

    },
    {
        'model':model3,
        'model_name':'model3',
        'epochs':30,
        'optimizer':keras.optimizers.Adam(learning_rate=0.1),
        'optimizer_name':'Adam',  
        'batch_size':64

    },
    {
        'model':model3,
        'model_name':'model3',
        'epochs':30,
        'optimizer':keras.optimizers.SGD(learning_rate=0.1),
        'optimizer_name':'SGD',  
        'batch_size':32

    }
    ]
    
    for i in range(len(settings)):
        epochs=settings[i]['epochs']
        batch_size=settings[i]['batch_size']
        optimizer=settings[i]['optimizer']
        optimizer_name=settings[i]['optimizer_name']
        model_name=settings[i]['model_name']
        model=settings[i]['model'] 
        history=training(model_name=model_name,
                         model=model,
                         epochs=epochs,
                         optimizer=optimizer,
                         batch_size=batch_size,
                         optimizer_name=optimizer_name,
                         x_train=x_train,
                         y_train=y_train,
                         x_val=x_val,
                         y_val=y_val)
        plot_save_charts(model_setting=settings[i],history=history)
        with open(f'models/{model_name}/model_result_{optimizer_name}__{batch_size}.json','w') as f:
            json.dump(history.history,f,indent=6)
            json.dump(history.params,f,indent=3)
        keras.backend.clear_session()
        model.reset_states()


