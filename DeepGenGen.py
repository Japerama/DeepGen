# Final Project - DeepGen
# CECS 590-50<br>
# Deep Learning Algorithms
# Author - Jon Harms
# Date Complete - 4/24/2020
# Due Date - Tuesday 4/24/2020

import os
import time
import numpy as np
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
from copy import copy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow_core.python.keras.layers import CuDNNLSTM
from rdkit import RDLogger, Chem
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.models import model_from_json

# Experiment 1 Parameters
# -> Files
experiment_description = "DeepGenExperiment_1"
dataset_file = "databases/Ligands/ligands.smi"
uncleaned_dataset_file = "NA"
ft_dataset_file = "NA"
# -> Training
# -> Generation parameters
# -> Checkpoint parameters
units = 256
data_length = 0
num_epochs = 20
ft_epochs = 0
batch_size = 20
ft_batch_size = 0
validation_split = 0.10
optimizer = "adam"
seed = 71
# -> Generation parameters
sampling_temp = 0.75
max_gen_length_smiles = 200
# -> Experiment Number Parameter
experiment_number = 1

# # Experiment 2 Parameters
# # -> Files
# experiment_description = "DeepGenExperiment_2"
# dataset_file = "databases/datasets/small_molecule_query_cleansed.smi"
# uncleaned_dataset_file = "databases/datasets/small_molecule_query.smi"
# ft_dataset_file = "databases/Ligands/ligands.smi"
# # -> Training
# units = 256
# data_length = 0
# num_epochs = 2
# ft_epochs = 10
# batch_size = 50
# ft_batch_size = 20
# validation_split = 0.10
# optimizer = "adam"
# seed = 71
# # -> Generation parameters
# sampling_temp = 0.75
# max_gen_length_smiles = 200
# # -> Experiment Number Parameter
# experiment_number = 2


# A class that performs the interpretation of a molecule SMILES string into a reference context for
# the Deep Learning model to ingest.
class SmilesInterpreter(object):
    def __init__(self):
        atoms = ['Al', 'As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'K', 'Li', 'N', 'Na', 'O', 'P', 'S', 'Se', 'Si',
                 'Te']  # possible atoms elements that can make up a SMILES string
        special = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '[', ']', '=', '#', '%',
                   '+', '-', 'se', 'te', 'c', 'n', 'o', 's', '@']  # possible special characters in a SMILES
        padding = ['G', 'E', 'A']  # G = start of a SMILES string, E = end, A = extra padding

        # Combines the characters into one table
        self.table = sorted(atoms, key=len, reverse=True) + special + padding
        table_len = len(self.table)

        # Seperate table characters into two lists with one containing characters with length of 2 and length of 1
        self.table_2_chars = list(filter(lambda x: len(x) == 2, self.table))
        self.table_1_chars = list(filter(lambda x: len(x) == 1, self.table))

        # Utilized as a lookup-table in one-hot-encoding techniques
        self.hot_encoding_dict = {}

        # Creates one-hot-encoding dictionary
        for i, symbol in enumerate(self.table):
            vec = np.zeros(table_len, dtype=np.float32)
            vec[i] = 1
            self.hot_encoding_dict[symbol] = vec

    # Function that interprets a SMILES string into a list of characters
    def interpret(self, smiles_string):
        smiles_string = smiles_string + ' '
        N = len(smiles_string)
        characters = []
        i = 0
        while (i < N):
            c1 = smiles_string[i]
            c2 = smiles_string[i:i + 2]
            if c2 in self.table_2_chars:
                characters.append(c2)
                i += 2
                continue
            if c1 in self.table_1_chars:
                characters.append(c1)
                i += 1
                continue
            i += 1
        return characters

    # Performs one-hot-encoding
    def hot_encoder(self, interpreted_smiles):
        result = np.array([self.hot_encoding_dict[symbol] for symbol in interpreted_smiles], dtype=np.float32)
        result = result.reshape(1, result.shape[0], result.shape[1])
        return result


# A class that prepares a set of data (in this case SMILES strings of molecules) for use within a Deep Learning Model
class DataPrep(Sequence):
    # Init function that prepares the DataPrep class based off data types - training data, validation data,
    # fine-tuning data
    def __init__(self, data_type='train'):
        self.data_type = data_type
        assert self.data_type in ['train', 'valid', 'finetune', 'valid_ft']
        self.max_len = 0

        # Loads the data
        if self.data_type == 'train':
            self.smiles_string = self._load(dataset_file)
        elif self.data_type == 'finetune':
            self.smiles_string = self._load(ft_dataset_file)
        else:
            pass

        # Sets up interpreter class for interpreting the incoming training data
        self.st = SmilesInterpreter()
        self.hot_encoding_dict = self.st.hot_encoding_dict

        # Interprets the incoming data
        self.interpreted_smiles = self._interpret(self.smiles_string)

        # If not fine-tuning data, splits the data between training and validation
        if self.data_type in ['train', 'valid', 'finetune', 'valid_ft']:
            self.idx = np.arange(len(self.interpreted_smiles))
            self.valid_size = int(np.ceil(len(self.interpreted_smiles) * validation_split))
            np.random.seed(seed)
            np.random.shuffle(self.idx)

    # Sets the data that belongs in each set
    def _set_data(self):
        if self.data_type == 'train':
            ret = [self.interpreted_smiles[self.idx[i]] for i in self.idx[self.valid_size:]]
        elif self.data_type == 'finetune':
            ret = [self.interpreted_smiles[self.idx[i]] for i in self.idx[self.valid_size:]]
        elif self.data_type == 'valid':
            ret = [self.interpreted_smiles[self.idx[i]] for i in self.idx[:self.valid_size]]
        elif self.data_type == 'valid_ft':
            ret = [self.interpreted_smiles[self.idx[i]] for i in self.idx[:self.valid_size]]
        else:
            ret = self.interpreted_smiles
        return ret

    # Loads the data
    def _load(self, data_filename):
        length = data_length
        print('Loading Molecules in the form of SMILES strings')
        with open(data_filename) as f:
            smiles_string = [s.rstrip() for s in f]
        if length != 0:
            smiles_string = smiles_string[:length]
        print('Completed')
        return smiles_string

    # Calls upon the interpreter class to interpret the imported SMILES strings.  Also updates the maximum
    # possible length of the training data.
    def _interpret(self, smiles_string):
        assert isinstance(smiles_string, list)
        print('Interpreting SMILES for Evaluation in Model')
        interpreted_smiles = [self.st.interpret(smi) for smi in tqdm(smiles_string)]
        if self.data_type == 'train':
            for interpreted_smi in interpreted_smiles:
                length = len(interpreted_smi)
                if self.max_len < length:
                    self.max_len = length
            max_gen_length_smiles = self.max_len
        elif self.data_type == 'finetune':
            for interpreted_smi in interpreted_smiles:
                length = len(interpreted_smi)
                if self.max_len < length:
                    self.max_len = length
            max_gen_length_smiles = self.max_len
        print('Completed')
        return interpreted_smiles

    #  Returns the amount of smiles per batch
    def __len__(self):
        target_interpreted_smiles = self._set_data()
        if self.data_type in ['train', 'valid']:
            ret = int(np.ceil(len(target_interpreted_smiles)/float(batch_size)))
        else:
            ret = int(np.ceil(len(target_interpreted_smiles)/float(ft_batch_size)))
        return ret

    # Provides particular characters back for the model
    def __getitem__(self, idx):
        target_interpreted_smiles = self._set_data()
        if self.data_type in ['train', 'valid']:
            data = target_interpreted_smiles[idx * batch_size:(idx + 1) * batch_size]
        else:
            data = target_interpreted_smiles[idx * ft_batch_size:(idx + 1) * ft_batch_size]
        data = self._padding(data)

        self.X, self.y = [], []
        for tp_smi in data:
            X = [self.hot_encoding_dict[symbol] for symbol in tp_smi[:-1]]
            self.X.append(X)
            y = [self.hot_encoding_dict[symbol] for symbol in tp_smi[1:]]
            self.y.append(y)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        return self.X, self.y, [None]

    # Adds padding when running the Deep Learning Algorithms of the model
    def _pad(self, interpreted_smi):
        return ['G'] + interpreted_smi + ['E'] + ['A' for _ in range(self.max_len - len(interpreted_smi))]

    # Returns the now-padded SMILES string
    def _padding(self, data):
        padded_smiles = [self._pad(t_smi) for t_smi in data]
        return padded_smiles


# Main Function, required for utilizing GPU Tensorflow in this application
if __name__ == '__main__':
    # Provides the number of GPUs that are available on the machine
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # Additional parameters necessary for utilzing GPU Tensorflow with CudnnLSTM
    multiprocessing.set_start_method('spawn')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    RDLogger.DisableLog('rdApp.*')

    # Creates an instance of the Smiles Interpreter that is utilized for importing and
    # analysis of the molecules via SMILES strings
    st = SmilesInterpreter()

    # Experiment 1
    if experiment_number == 1:
        # Creates the Expirement and Plugin directories, Tensorboard and Checkpoint directories
        exp_dir = os.path.join('experiments', time.strftime('%Y-%m-%d/', time.localtime()), experiment_description)
        checkpoint_dir = os.path.join('experiments', time.strftime('%Y-%m-%d/', time.localtime()),
                                      experiment_description,
                                      'checkpoints/')
        dirs = [exp_dir, checkpoint_dir]

        # Creates the directories
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        # Assigns the training data and validation data from the original training set
        train_dl = DataPrep(data_type='train')
        valid_dl = copy(train_dl)
        valid_dl.data_type = 'valid'

        # Defines the input weights and shape to the model
        # Input shape = SMILES tokens
        # Input weights = normalized data
        n_table = len(st.table)
        weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)

        # Creates the Sequential model for original training set
        # Two CuDNNLSTM network layers, following by a Dense prediction layer
        model = Sequential()
        model.add(CuDNNLSTM(units=units,
                            input_shape=(None, n_table),
                            return_sequences=True,
                            kernel_initializer=weight_init))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(units=units,
                            input_shape=(None, n_table),
                            return_sequences=True,
                            kernel_initializer=weight_init))
        model.add(Dropout(0.2))
        model.add(Dense(units=n_table,
                        activation='softmax',
                        kernel_initializer=weight_init))

        # Compiles the model, utilizing accuracy as the metrics to track during epochs
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Sets up the callbacks information to be utilized during training of the model
        callbacks = [ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % experiment_description),
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            save_weights_only=True,
            verbose=2, ),
            # TensorBoard(log_dir=tensorboard_log_dir, write_graph=True, )
            ]

        print(model.summary())

        # Trains the model on the training and validation data, utilizing specified parameters
        start = time.perf_counter()
        history = model.fit(
            train_dl,
            steps_per_epoch=train_dl.__len__(),
            epochs=num_epochs,
            verbose=True,
            validation_data=valid_dl,
            validation_steps=valid_dl.__len__(),
            use_multiprocessing=True,
            shuffle=True,
            callbacks=callbacks)
        elapsed_Ligand_gen = time.perf_counter() - start
        print('Total time spent training the model is %.3f seconds.' % elapsed_Ligand_gen)

        print(history.history.keys())

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('Ligand_gen Accuracy')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('Ligand_gen Val_Loss')
        plt.show()

        final_accuracy_Ligand_gen = history.history['val_accuracy'][-1]
        final_val_loss_Ligand_gen = history.history['val_loss'][-1]

        # Utilizing the trained model, the following code generates 100 novel molecules sequentially,
        # which are represented as SMILES strings. These strings are then appended to an array of strings.
        # Shows a progress bar.
        start = 'G'
        num = 100
        Ligand_gen_smiles = []
        for _ in tqdm(range(num)):
            sequence = start
            while (len(st.interpret(sequence)) <= max_gen_length_smiles) and (sequence[-1] != 'E'):
                x = st.hot_encoder(st.interpret(sequence))
                preds = model.predict_on_batch(x)[0][-1]
                streched = np.log(preds) / sampling_temp
                streched_probs = np.exp(streched) / np.sum(np.exp(streched))
                next_character = np.random.choice(range(len(streched)), p=streched_probs)
                sequence += st.table[next_character]
            sequence = sequence[1:].rstrip('E')
            Ligand_gen_smiles.append(sequence)

        # Outputs the generated molecules to a configuration file which is used for metrics calculations in
        # an iPython Notebook
        new_file = os.path.join('databases/output_molecules/Ligand_gen_smiles', 'Ligand_gen_smiles.smi')
        fi = open(new_file, "w")
        for smi in Ligand_gen_smiles:
            fi.write(str(smi) + '\n')
        fi.close()
        print('Generated Ligand_gen_smiles file')

        # RDKit is utilized to produce the validity of a molecule, if not valid, it will not be added to the
        # valid molecules list
        valid_mols = []
        for smi in Ligand_gen_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)
        Ligand_gen_molecules_validity = len(valid_mols) / num
        print(Ligand_gen_molecules_validity)

        # RDKit is utilized to produce the uniqueness of a molecule to the other molecules in the array
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        Ligand_gen_molecules_uniqueness = len(set(valid_smiles)) / len(valid_smiles)
        print(Ligand_gen_molecules_uniqueness)

        print(model.summary())

        # Writes metrics data from both original and fine-tuned models to a metrics configuration file.
        # This file is used in an external iPython notebook for evaluation.
        metrics_file = 'Ligand_metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write('%f\n' % Ligand_gen_molecules_validity)
            f.write('%f\n' % Ligand_gen_molecules_uniqueness)
            f.write('%f\n' % batch_size)
            f.write('%f\n' % num_epochs)
            f.write('%.3f\n' % elapsed_Ligand_gen)
            f.write('%.3f\n' % final_val_loss_Ligand_gen)
            f.write('%.3f\n' % final_accuracy_Ligand_gen)
        f.close()
    else:  # Experiment 2
        # Ensures the original dataset file has not been cleansed yet
        assert os.path.exists(uncleaned_dataset_file)
        assert not os.path.exists(dataset_file), f'{dataset_file} already exists.'

        # Opens the uncleansed dataset file and then cleans the data within. Then creates a new file
        # containing the cleansed data.
        with open(uncleaned_dataset_file, 'r') as f:
            smiles = [l.rstrip() for l in f]
        print(f'input SMILES num: {len(smiles)}')
        print('Cleansing begins')
        cleansed_smiles = list(set([s for s in smiles if s]))
        print('done.')
        print(f'output SMILES num: {len(cleansed_smiles)}')
        with open(dataset_file, 'w') as f:
            for smi in cleansed_smiles:
                f.write(smi + '\n')
        print('File cleansed and output.')

        # Creates the Expirement and Plugin directories, Tensorboard and Checkpoint directories
        exp_dir = os.path.join('experiments', time.strftime('%Y-%m-%d/', time.localtime()), experiment_description)
        checkpoint_dir = os.path.join('experiments', time.strftime('%Y-%m-%d/', time.localtime()),
                                      experiment_description,
                                      'checkpoints/')
        dirs = [exp_dir, checkpoint_dir]

        # Creates the directories
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        # Assigns the training data and validation data from the original training set
        train_dl = DataPrep(data_type='train')
        valid_dl = copy(train_dl)
        valid_dl.data_type = 'valid'

        # Defines the input weights and shape to the model
        # Input shape = SMILES tokens
        # Input weights = normalized data
        n_table = len(st.table)
        weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)

        # Creates the Sequential model for original training set
        # Two CuDNNLSTM network layers, following by a Dense prediction layer
        model = Sequential()
        model.add(CuDNNLSTM(units=units,
                            input_shape=(None, n_table),
                            return_sequences=True,
                            kernel_initializer=weight_init))
        model.add(Dropout(0.2)),
        model.add(CuDNNLSTM(units=units,
                            input_shape=(None, n_table),
                            return_sequences=True,
                            kernel_initializer=weight_init))
        model.add(Dropout(0.2)),
        model.add(Dense(units=n_table,
                        activation='softmax',
                        kernel_initializer=weight_init))

        # Compiles the model, utilizing accuracy as the metrics to track during epochs
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        # Sets up the callbacks information to be utilized during training of the model
        callbacks = [ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % experiment_description),
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            save_weights_only=True,
            verbose=2, ),
        ]

        # Trains the model on the training and validation data, utilizing specified parameters
        start = time.perf_counter()
        history = model.fit(
            train_dl,
            steps_per_epoch=train_dl.__len__(),
            epochs=num_epochs,
            verbose=True,
            validation_data=valid_dl,
            validation_steps=valid_dl.__len__(),
            use_multiprocessing=True,
            shuffle=True,
            callbacks=callbacks)
        elapsed_origin_gen = time.perf_counter() - start
        print('Total time spent training the model is %.3f seconds.' % elapsed_origin_gen)

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")

        print(history.history.keys())

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('Origin_gen Val_Loss')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('Origin_gen Val_Loss')
        plt.show()

        final_accuracy_origin_gen = history.history['val_accuracy'][-1]
        final_val_loss_origin_gen = history.history['val_loss'][-1]

        # Utilizing the trained model, the following code generates 100 novel molecules sequentially,
        # which are represented as SMILES strings. These strings are then appended to an array of strings.
        # Shows a progress bar.
        start = 'G'
        num = 100
        origin_gen_smiles = []
        for _ in tqdm(range(num)):
            sequence = start
            while (len(st.interpret(sequence)) <= max_gen_length_smiles) and (sequence[-1] != 'E'):
                x = st.hot_encoder(st.interpret(sequence))
                preds = model.predict_on_batch(x)[0][-1]
                streched = np.log(preds) / sampling_temp
                streched_probs = np.exp(streched) / np.sum(np.exp(streched))
                next_character = np.random.choice(range(len(streched)), p=streched_probs)
                sequence += st.table[next_character]
            sequence = sequence[1:].rstrip('E')
            origin_gen_smiles.append(sequence)

        # Outputs the generated molecules to a configuration file which is used for metrics calculations in
        # an iPython Notebook
        new_file = os.path.join('databases/output_molecules/Origin_gen_smiles', 'Origin_gen_smiles.smi')
        fi = open(new_file, "w")
        for smi in origin_gen_smiles:
            fi.write(str(smi) + '\n')
        fi.close()
        print('Made Origin_gen_smiles file')

        # RDKit is utilized to produce the validity of a molecule, if not valid, it will not be added to the
        # valid molecules list
        valid_mols = []
        for smi in origin_gen_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)
        origin_gen_molecules_validity = len(valid_mols) / num
        print(origin_gen_molecules_validity)

        # RDKit is utilized to produce the uniqueness of a molecule to the other molecules in the array
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        origin_gen_molecules_uniqueness = len(set(valid_smiles)) / len(valid_smiles)
        print(origin_gen_molecules_uniqueness)

        # Beginning of Fine-tuning Section

        # json_file = open('model.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # # load weights into new model
        # loaded_model.load_weights("model.h5")

        model_old = model

        print(model_old.summary())

        model_old.pop()

        model_old.outputs = [model_old.layers[-1].output]

        for layer in model_old.layers:
            layer.trainable = False

        # Assigns the training data and validation data from the fine-tuning set
        finetune_dl = DataPrep(data_type='finetune')
        ft_valid_dl = copy(finetune_dl)
        ft_valid_dl.data_type = 'valid_ft'

        # Defines the input weights and shape to the model
        # Input shape = SMILES tokens
        # Input weights = normalized data
        n_table = len(st.table)
        weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)

        callbacks_ft = [ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'ft%s-{epoch:02d}-{val_loss:.2f}.hdf5' % experiment_description),
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            save_weights_only=True,
            verbose=2, ),
        ]

        # Creates the Transfer Learning Fine-Tuning Sequential model for fine-tuning set
        # The Original Model + Two new CuDNNLSTM network layers, following by a new Dense prediction layer
        model_new = Sequential([
            model_old,
            CuDNNLSTM(units=units,
                      input_shape=(None, n_table),
                      return_sequences=True,
                      kernel_initializer=weight_init),
            Dropout(0.2),
            CuDNNLSTM(units=units,
                      input_shape=(None, n_table),
                      return_sequences=True,
                      kernel_initializer=weight_init),
            Dropout(0.2),
            Dense(units=n_table,
                  activation='softmax',
                  kernel_initializer=weight_init)])

        print(model_new.summary())

        # Compiles the model, utilizing accuracy as the metrics to track during epochs
        model_new.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Trains the model on the training and validation data, utilizing specified parameters
        start = time.perf_counter()
        history = model_new.fit(
            finetune_dl,
            steps_per_epoch=finetune_dl.__len__(),
            epochs=ft_epochs,
            verbose=True,
            validation_data=ft_valid_dl,
            validation_steps=ft_valid_dl.__len__(),
            use_multiprocessing=True,
            shuffle=True,
            callbacks=callbacks_ft)
        elapsed_origin_ft_gen = time.perf_counter() - start
        print('Total time spent training the model is %.3f seconds.' % elapsed_origin_ft_gen)

        print(history.history.keys())

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('Origin_FT_gen Val_Loss')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('Origin_FT_gen Val_Loss')
        plt.show()

        final_accuracy_origin_ft_gen = history.history['val_accuracy'][-1]
        final_val_loss_origin_ft_gen = history.history['val_loss'][-1]

        # Utilizing the fine-tuned trained model, the following code generates 100 novel molecules sequentially,
        # which are represented as SMILES strings. These strings are then appended to an array of strings.
        # Shows a progress bar.
        num = 100
        start = 'G'
        origin_ft_smiles = []
        for _ in tqdm(range(num)):
            sequence = start
            while (len(st.interpret(sequence)) <= max_gen_length_smiles) and (sequence[-1] != 'E'):
                x = st.hot_encoder(st.interpret(sequence))
                preds = model_new.predict_on_batch(x)[0][-1]
                streched = np.log(preds) / sampling_temp
                streched_probs = np.exp(streched) / np.sum(np.exp(streched))
                next_character = np.random.choice(range(len(streched)), p=streched_probs)
                sequence += st.table[next_character]
            sequence = sequence[1:].rstrip('E')
            origin_ft_smiles.append(sequence)

        # Outputs the generated molecules to a configuration file which is used for metrics calculations in
        # an iPython Notebook
        new_file = os.path.join('databases/output_molecules/Origin_ft_gen_smiles', 'Origin_ft_gen_smiles.smi')
        fi = open(new_file, "w")
        for smi in origin_ft_smiles:
            fi.write(str(smi) + '\n')
        fi.close()
        print('Made Origin_gen_Finetuned_SMILES file')

        # RDKit is utilized to produce the validity of a molecule, if not valid, it will not be added to the
        # valid molecules list
        valid_fine_tuned_mols = []
        for smi in origin_ft_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_fine_tuned_mols.append(mol)
        origin_ft_gen_molecules_validity = len(valid_fine_tuned_mols) / num
        print(origin_ft_gen_molecules_validity)

        # RDKit is utilized to produce the uniqueness of a molecule to the other molecules in the array
        valid_fine_tuned_smiles = [Chem.MolToSmiles(mol) for mol in valid_fine_tuned_mols]
        origin_ft_gen_molecules_uniqueness = len(set(valid_fine_tuned_smiles)) / len(valid_fine_tuned_smiles)
        print(origin_ft_gen_molecules_uniqueness)

        # Writes metrics data from both original and fine-tuned models to a metrics configuration file.
        # This file is used in an external iPython notebook for evaluation.
        metrics_file = 'origin_metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write('%f\n' % origin_gen_molecules_validity)
            f.write('%f\n' % origin_ft_gen_molecules_validity)
            f.write('%f\n' % origin_gen_molecules_uniqueness)
            f.write('%f\n' % origin_ft_gen_molecules_uniqueness)
            f.write('%f\n' % batch_size)
            f.write('%f\n' % ft_batch_size)
            f.write('%f\n' % num_epochs)
            f.write('%f\n' % ft_epochs)
            f.write('%.3f\n' % elapsed_origin_gen)
            f.write('%.3f\n' % elapsed_origin_ft_gen)
            f.write('%.3f\n' % final_val_loss_origin_gen)
            f.write('%.3f\n' % final_val_loss_origin_ft_gen)
            f.write('%.3f\n' % final_accuracy_origin_gen)
            f.write('%.3f\n' % final_accuracy_origin_ft_gen)
        f.close()
