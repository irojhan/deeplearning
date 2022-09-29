# import tensorflow
import tensorflow as tf
from tensorflow.keras import *

# import tensorflow datasets
import tensorflow_datasets as TFDatasets

"""
-----------------------------------------------------------------------------------------------
Step 1: Load dataset spoken_digit [30 points]
Load dataset from Tensorflow Datasets;
Set batch size to 64;
Preprocess the dataset to get spectogram by applying the short-time Fourier transform; 
Split the dataset with 70% of training, 10% of validation, and 20% of testing;
-----------------------------------------------------------------------------------------------
"""

# load mnist dataset
dataset = 

# initialize zero padding function and fourier transform
def stft(waveform: tf.Tensor, label: int) -> tuple[tf.Tensor, int]:
    spectrogram = 
    label = 
    return spectrogram, label

# mapping to dataset
dataset = 

# split dataset
training_dataset = 
validation_dataset = 
testing_dataset =

"""
-----------------------------------------------------------------------------------------------
Step 2: Define CRNN model [30 points]
1. Resize layer to resize the spectrogram to 32x32
2. Normalization layer to normalize the input data based on its mean and standard deviation
3. Conv2D layer named 'conv1' with 64 filters, 3x3 kernel size, same padding, and relu activation
4. BatchNormalization layer to normalize axis 3
5. MaxPooling2D layer to pool the features
6. Conv2D layer named 'conv2' with 64 filters, 3x3 kernel size, same padding, and relu activation
7. BatchNormalization layer to normalize axis 3
8. MaxPooling2D layer to pool the features
9. Permute layer to permute the frequency axis and time axis
10. Reshape the permuted output to (-1, shape[1], shape[2] * shape[3])
11. A GRU layer named 'gru1' with 512 units
12. A GRU layer named 'gru2' with 512 units
13. A Dropout layer with dropout ratio 0.5
14. A Dense layer to do the classification
-----------------------------------------------------------------------------------------------
"""

# wrap to model
input_data = 
y = 
model = Model(name='spoken_digit_classification')

"""
-----------------------------------------------------------------------------------------------
Step 3: Compile the model [10 points]
Adam optimizer with 0.001 learning rate, multiplies 0.9 for each epoch;
Categorical crossentropy as loss function;
Accuracy as the metrics;
-----------------------------------------------------------------------------------------------
"""

model.compile()

"""
-----------------------------------------------------------------------------------------------
Step 4: Train the model with training dataset [10 points]
Set epoch size to 30;
TensorBoard Callback to record the metrics for each epoch;
Checkpoint Callback checkpoints;
-----------------------------------------------------------------------------------------------
"""

model.fit()

"""
-----------------------------------------------------------------------------------------------
Step 5: Evaluate the model with testing dataset [10 points]
-----------------------------------------------------------------------------------------------
"""

model.evaluate()


"""
-----------------------------------------------------------------------------------------------
Step 6: Remember to submit results to Canvas. [10 points]
A screenshot of TensorBoard;
A screenshot of the training and testing procedure;
-----------------------------------------------------------------------------------------------
"""
