# Import
import numpy as np
import os
import gc
import tensorflow as tf
import random
from tqdm import tqdm
import keras.backend as K

import tensorflow_addons as tfa

import warnings
warnings.filterwarnings('ignore')
from types import SimpleNamespace

import argparse

import wandb


print('GPU is', 'available' if tf.config.list_physical_devices('GPU') else 'NOT AVAILABLE')

tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()

configs = SimpleNamespace(
    learning_rate = 0.001,
    num_steps = 0.9,
    scheduler = '',
    exp = '',
    start_batch = 1,
    max_files = 330,
    epochs = 30,

)

# Training
valid_dataset = [1,2,3]
validation_files_amount = len(valid_dataset)
batch_size = 2048               # for A5000: 2048
verbose = 0
n_files = 50
batch_file_len = 200_000



# Model Parameters
pulse_count = 96
feature_count = 8
lstm_units = 196 
bin_num = 96 // 4


# Data
base_dir = "tf_dataset/"
file_format = base_dir + 'aux_pointpicker_mp96_n11_batch_{batch_id:d}.npz'


# Set Seed
seed = 4242
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)


from tensorflow.keras.losses import categorical_crossentropy

def scce_with_ls(y, y_hat):
    y = tf.one_hot(tf.cast(y, tf.int32), bin_num * bin_num)
    return categorical_crossentropy(y, y_hat, label_smoothing = 0.01)


def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse 
    cosine (arccos) thereof is then the angle between the two input vectors
    
    Parameters:
    -----------
    
    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian
    
    Returns:
    --------
    
    dist : float
        mean over the angular distance(s) in radian
    '''
    
    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")
    
    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod =  np.clip(scalar_prod, -1, 1)
    
    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))


# Create Azimuth Edges
azimuth_edges = np.linspace(0, 2 * np.pi, bin_num + 1)
print(azimuth_edges)

# Create Zenith Edges
zenith_edges = []
zenith_edges.append(0)
for bin_idx in range(1, bin_num):
    zenith_edges.append(np.arccos(np.cos(zenith_edges[-1]) - 2 / (bin_num)))
zenith_edges.append(np.pi)
zenith_edges = np.array(zenith_edges)
print(zenith_edges)

angle_bin_zenith0 = np.tile(zenith_edges[:-1], bin_num)
angle_bin_zenith1 = np.tile(zenith_edges[1:], bin_num)
angle_bin_azimuth0 = np.repeat(azimuth_edges[:-1], bin_num)
angle_bin_azimuth1 = np.repeat(azimuth_edges[1:], bin_num)

angle_bin_area = (angle_bin_azimuth1 - angle_bin_azimuth0) * (np.cos(angle_bin_zenith0) - np.cos(angle_bin_zenith1))
angle_bin_vector_sum_x = (np.sin(angle_bin_azimuth1) - np.sin(angle_bin_azimuth0)) * ((angle_bin_zenith1 - angle_bin_zenith0) / 2 - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4)
angle_bin_vector_sum_y = (np.cos(angle_bin_azimuth0) - np.cos(angle_bin_azimuth1)) * ((angle_bin_zenith1 - angle_bin_zenith0) / 2 - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4)
angle_bin_vector_sum_z = (angle_bin_azimuth1 - angle_bin_azimuth0) * ((np.cos(2 * angle_bin_zenith0) - np.cos(2 * angle_bin_zenith1)) / 4)

angle_bin_vector_mean_x = angle_bin_vector_sum_x / angle_bin_area
angle_bin_vector_mean_y = angle_bin_vector_sum_y / angle_bin_area
angle_bin_vector_mean_z = angle_bin_vector_sum_z / angle_bin_area

angle_bin_vector = np.zeros((1, bin_num * bin_num, 3))
angle_bin_vector[:, :, 0] = angle_bin_vector_mean_x
angle_bin_vector[:, :, 1] = angle_bin_vector_mean_y
angle_bin_vector[:, :, 2] = angle_bin_vector_mean_z

def pred_to_angle(pred, epsilon=1e-8):
    # convert prediction to vector
    pred_vector = (pred.reshape((-1, bin_num * bin_num, 1)) * angle_bin_vector).sum(axis=1)
    
    # normalize
    pred_vector_norm = np.sqrt((pred_vector**2).sum(axis=1))
    mask = pred_vector_norm < epsilon
    pred_vector_norm[mask] = 1
    
    # assign <1, 0, 0> to very small vectors (badly predicted)
    pred_vector /= pred_vector_norm.reshape((-1, 1))
    pred_vector[mask] = np.array([1., 0., 0.])
    
    # convert to angle
    azimuth = np.arctan2(pred_vector[:, 1], pred_vector[:, 0])
    azimuth[azimuth < 0] += 2 * np.pi
    zenith = np.arccos(pred_vector[:, 2])
    
    return azimuth, zenith

def y_to_angle_code(batch_y):
    azimuth_code = (batch_y[:, 0] > azimuth_edges[1:].reshape((-1, 1))).sum(axis=0)
    zenith_code = (batch_y[:, 1] > zenith_edges[1:].reshape((-1, 1))).sum(axis=0)
    angle_code = bin_num * azimuth_code + zenith_code
    
    return angle_code

def normalize_data(data):
    data[:, :, 0] /= 1000   # time
    data[:, :, 1] /= 300    # charge
    data[:, :, 3:] /= 600   # x
    # data[:, :, 4] /= 600   # y
    # data[:, :, 5] /= 600   # z
    
    return data

def prep_validation_data(valid_dataset):
    print("Processing Validation Data...")

    # Prepare fixed Validation Set
    val_x = None
    val_y = None
    
    # Summary
    print(valid_dataset)

    # Loop
    for batch_id in tqdm(valid_dataset):
        val_data_file = np.load(file_format.format(batch_id = batch_id))

        if val_x is None:
            val_x = val_data_file["x"][:, :, [0,1,2,3,4,5,6,7,]] # 6, 7]]
            val_y = val_data_file["y"]
        else:
            val_x = np.append(val_x, val_data_file["x"][:, :, [0,1,2,3,4,5,6,7,]], axis = 0) # 6, 7]], axis = 0)
            val_y = np.append(val_y, val_data_file["y"], axis = 0)

        val_data_file.close()
        del val_data_file
        _ = gc.collect()

    # Normalize Data
    val_x = normalize_data(val_x)

    # Shape Summary
    print(val_x.shape)
    
    return val_x, val_y

def prep_training_data(start_batch, end_batch):
    print("Processing Training Data...")
    
    # Placeholders
    train_x = None
    train_y = None

    # Summary
    print(train_batch_ids[start_batch:end_batch])
    
    # Loop
    for batch_id in tqdm(train_batch_ids[start_batch:end_batch]):
        train_data_file = np.load(file_format.format(batch_id = batch_id))

        if train_x is None:
            train_x = train_data_file["x"][:, :, [0,1,2,3,4,5,6,7,]] # 6, 7]]
            train_y = train_data_file["y"]
        else:
            train_x = np.append(train_x, train_data_file["x"][:, :, [0,1,2,3,4,5,6,7,]], axis = 0) #6,7]], axis = 0)
            train_y = np.append(train_y, train_data_file["y"], axis = 0)

        train_data_file.close()
        del train_data_file
        _ = gc.collect()

    # Normalize data
    train_x = normalize_data(train_x)
    
    # Shape Summary
    print(train_x.shape)
    
    # Output Encoding
    trn_y_anglecode = y_to_angle_code(train_y)
        
    return train_x, trn_y_anglecode


def create_model(args):
    with strategy.scope(): 
        model = tf.keras.models.Sequential([
            tf.keras.layers.Masking(mask_value = 0., input_shape = (pulse_count, feature_count)),
           # tf.keras.layers.GaussianNoise(stddev=0.01),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(lstm_units, return_sequences = True, dropout= 0.0)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(lstm_units, return_sequences = True, dropout=0.0)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(lstm_units, return_sequences = True, dropout=0.0)),
           # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(lstm_units, return_sequences = True, dropout=0.0, name ='gru3a'), name = 'ss3a'),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(lstm_units, dropout=0.0,)),
            tf.keras.layers.Dense(256, activation = 'relu',),
            tf.keras.layers.Dense(bin_num * bin_num, activation='softmax',),
        ])
        

        

        if args.scheduler == 'cosine':            
            decay_steps = total_number_of_steps * configs.num_steps
            cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate = configs.learning_rate,
                decay_steps = decay_steps,
                alpha=0.1)
            optimizer= tf.keras.optimizers.Adam(cosine_decay_scheduler)
        else:
            print('No scheduler and learining rate is set to: ', configs.learning_rate, '')
            optimizer= tf.keras.optimizers.Adam(learning_rate = configs.learning_rate) #(learning_rate = learning_rate)
            #optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate = learning_rate, weight_decay = 0.005, amsgrad=True)

        # Lookahead
        #optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=6, slow_step_size=0.5)

        loss = scce_with_ls 

        model.compile(loss = loss, #'sparse_categorical_crossentropy',
                      optimizer= optimizer,
                      metrics = ['accuracy'])
        
        model.summary()

        return model




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_units', type=int, default=196)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_steps', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--scheduler', type=str, default='')
    parser.add_argument('--start_batch', type=int, default=4)
    parser.add_argument('--max_files', type=int, default=330)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_chunks', type=int, default=0)

    args = parser.parse_args()
    vars(configs).update(vars(args))
    return args

if __name__ == '__main__':
    args = parse_args()
    # print values from args
    print('*' * 20)
    print(args)
    print('*' * 20) 
    
    train_batch_id_min = args.start_batch #331
    max_files = args.max_files # 330 #320
    train_batch_ids = [*range(train_batch_id_min, train_batch_id_min+max_files+1)]
    total_number_of_steps = (batch_file_len * max_files * args.epochs) // batch_size
    print(train_batch_ids)     

    wandb.init(project="icecube-new-big", name=args.exp, config=configs)
    # Get Fixed Validation Dataset
    val_x, val_y = prep_validation_data(valid_dataset)


    model = create_model(args)
    
    if configs.resume != '':
        model = tf.keras.models.load_model(configs.resume, compile = True)
        print('Model loaded from: ', configs.resume)
        K.set_value(model.optimizer.lr, configs.learning_rate)


    chunks = ((max_files) // n_files)

    remainder = (max_files) % n_files
    if remainder != 0:
        chunks += 1

    print(f'Chunks: {chunks} Remainder: {remainder}')

    # Epoch Loop
    for e in range(args.epochs):
        print(f'=========== EPOCH: {e}')
        

        for n_chunk in range(chunks):
            print(f'=========== CHUNK: {n_chunk}')
            start_batch = n_chunk * n_files

            if n_chunk == chunks - 1:
                end_batch = start_batch + remainder
            else:
                end_batch = start_batch + n_files 
            
            print(f' start_batch: {start_batch} end_batch: {end_batch}')
            trn_x, trn_y_anglecode = prep_training_data(start_batch, end_batch)


            batch_count = trn_x.shape[0] // batch_size
            print(f'Number of batches: {batch_count}')

            indices = np.arange(trn_x.shape[0])
            np.random.shuffle(indices)
            trn_x = trn_x[indices]
            trn_y_anglecode = trn_y_anglecode[indices]
                
            # Placeholder
            losses = []
            accuracy = []
                
            # Batch Loop
            for batch_index in tqdm(range(batch_count), total = batch_count):
                b_train_x = trn_x[batch_index * batch_size: batch_index * batch_size + batch_size,:]
                b_train_y = trn_y_anglecode[batch_index * batch_size: batch_index * batch_size + batch_size]
                
                metrics = model.train_on_batch(b_train_x, b_train_y)
                losses.append(metrics[0])
                accuracy.append(metrics[1])

                wandb.log({"train loss": metrics[0]})
                wandb.log({"train accuracy": metrics[1]})  
                wandb.log({"learning rate": K.eval(model.optimizer.lr)})
            
            if args.save_chunks:
                
                valid_pred = model.predict(val_x, batch_size = batch_size, verbose = verbose)    
                valid_pred_azimuth, valid_pred_zenith = pred_to_angle(valid_pred)
                mae = angular_dist_score(val_y[:, 0], val_y[:, 1], valid_pred_azimuth, valid_pred_zenith)   
                model.save(f'mae_{mae:.5f}-chunk-{n_chunk}-{args.exp}_pp96_n{feature_count}_bin{bin_num}_batch{batch_size}_epoch{e}.h5', include_optimizer = True) 
                print(f'CHUNK ->>>> Total Train Loss: {np.mean(losses):.4f}   Accuracy: {np.mean(accuracy):.4f}  MAE: {mae:.5f}')
                wandb.log({"val chunk loss": np.mean(losses)})
                wandb.log({"val chunk accuracy": np.mean(accuracy)}) 
                wandb.log({"chunk mae_loss": mae})  
        #Save Model
        model.save(f'{args.exp}_pp96_n{feature_count}_bin{bin_num}_batch{batch_size}_epoch{e}.h5', include_optimizer = True)

        # Metrics
        valid_pred = model.predict(val_x, batch_size = batch_size, verbose = verbose)    
        valid_pred_azimuth, valid_pred_zenith = pred_to_angle(valid_pred)
        mae = angular_dist_score(val_y[:, 0], val_y[:, 1], valid_pred_azimuth, valid_pred_zenith)    
        print(f'Total Train Loss: {np.mean(losses):.4f}   Accuracy: {np.mean(accuracy):.4f}  MAE: {mae:.5f}')  
        wandb.log({"val loss": np.mean(losses)})
        wandb.log({"val accuracy": np.mean(accuracy)}) 
        wandb.log({"mae_loss": mae}) 
        # Memory Cleanup
        gc.collect()
