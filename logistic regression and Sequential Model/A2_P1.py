import tensorflow as tf
from tensorflow.keras import *

"""
-----------------------------------------------------------------------------------------------
Step 1: Load dataset MNIST [10 points]
Load dataset from Keras;
Wrap to tensorflow dataset;
Set epoch size to 20;
Set batch size to 256 (or fit your GPU GRAM size);
-----------------------------------------------------------------------------------------------
"""

(x_train, y_train), (x_test, y_test) = _________________________________

# data normalization
__________________________________________
__________________________________________
__________________________________________
__________________________________________

training_dataset = __________________________________________
testing_dataset = __________________________________________

"""
-----------------------------------------------------------------------------------------------
Step 2: Define logistic regression model [15 points]
Initialize the model instance;
Use CategoricalCrossentropy as your loss function;
Use CategoricalAccuracy as your metrics;
Use Adam with its default settings as your optimizer;
-----------------------------------------------------------------------------------------------
"""

class LogisticRegression(tf.Module):
    W: tf.Variable
    b: tf.Variable

    \\TODO: define the __init__ method and complete the forward pass (__call__)

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        _________________________________
        _________________________________
        _________________________________
        y: tf.Tensor =
        return y
 
model = _________________________________
loss_fn = _________________________________
acc_fn = _________________________________
optimizer = _________________________________

"""
-----------------------------------------------------------------------------------------------
Step 3: train the model [20 points]
Use summrary writer record TensorBoard data;
Use Checkpoint to save checkpoints;
Use training_dataset to train the model;
-----------------------------------------------------------------------------------------------
"""

train_summary_writer = _________________________________
ckpt_path = "ckpt/checkpoint"

# use enumerate to iterate over data  
for _________________________________:
    with tf.GradientTape() as tape:
        y = _________________________________
        loss: tf.Tensor = _________________________________
        acc: tf.Tensor = _________________________________
        
    grads: list[tf.Tensor] = _________________________________
    optimizer.apply_gradients(_________________________________)
    
    if _________________________________ : # the number of iterations meets a requirement
        print("Iteration %d: loss=%f, accuracy=%f" % (_____, loss, acc)) # iteration
        with train_summary_writer.as_default():
            tf.summary.scalar(_________________________________)
            tf.summary.scalar(_________________________________)
    
    if _________________________________:# the number of iterations meets a requirement
        checkpoint =_________________________________
        checkpoint.write(ckpt_path)

"""
-----------------------------------------------------------------------------------------------
Step 4: Test the model [15 points]
Record each loss and accuracy in the list;
Calculate the mean loss and accuracy;
Hint: If the accuracy is less then 85%, increase your epoch size and use a different random function for the weights.
Final accuracy = _________________________________
-----------------------------------------------------------------------------------------------
"""

loss_list: list[tf.Tensor] = list()
accuracy_list: list[tf.Tensor] = list()

for _________________________________:
    y = _________________________________
    l = _________________________________
    a = _________________________________
    _________________________________
    _________________________________
    
loss_eval = _________________________________
acc_eval = _________________________________
tf.print("eval_loss=%f, eval_acc=%f" % (loss_eval, acc_eval))

"""
-----------------------------------------------------------------------------------------------
Remember to submit results to Canvas. [10 points]
-----------------------------------------------------------------------------------------------
"""
