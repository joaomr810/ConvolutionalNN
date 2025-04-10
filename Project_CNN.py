#!/usr/bin/env python
# coding: utf-8

# # Intel Image Classification
# 
# The Intel Image Classification dataset is a collection of labeled images categorized into 6 types: **buildings, forest, glacier, mountain, sea, and street**. This dataset is intended for image classification tasks, providing a variety of real-world environmental categories. It can be used to train machine learning models for visual recognition tasks, helping in the development of models for image classification, especially in areas like environmental monitoring or geographical mapping.

# ## Required installations and libraries
# 
# Can be installed using the command `! pip install` below or by executing the installation with the `requirements.txt` file in the terminal using the command `pip install -r requirements.txt`.

# In[ ]:


# ! pip install numpy
# ! pip install pandas
# ! pip install matplotlib
# ! pip install tensorflow


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from collections import Counter
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import split_dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import load_model


# ## Auxiliary code/functions

# In[2]:


def counting_iterator(dataset_iterator, dataset_name, item_type='images'):
    """
    Counts the number of items (images or batches) in a TensorFlow dataset iterator.
    Prints the result as a message without returning anything.

    Input:
        dataset_iterator: tf.data.Dataset
            A TensorFlow dataset iterator containing the data.
        dataset_name: str
            Name of the dataset (e.g., 'train', 'test', 'validation').
        item_type: str, optional
            Specifies whether we are counting individual 'images' or 'batches'. Defaults to 'images'.

    Output:
        None
    """
    total_items = sum(1 for _ in dataset_iterator)

    print(f'The {dataset_name} dataset has {total_items} {item_type}.')


# In[3]:


def count_items_by_label(dataset, labels_names):
    """
    Counts the number of items per label in a dataset.

    Input:
        dataset: tf.data.Dataset
            A TensorFlow dataset iterator containing the data.
        labels_names: list
            A list of class labels corresponding to the indices in the dataset.

    Output:
        label_counting: dict
            A dictionary with the class index as keys and the number of items as values.
    """

    label_counting = Counter()

    for _, label in dataset:
        class_idx = np.argmax(label.numpy())    # converting the tensor label to a numpy array and obtaining the index with argmax
        label_counting[class_idx] += 1          # label_counting will be a dictionary with label_idx as keys and number of images as values
    
    for index, count in label_counting.items():
        print(f'{labels_names[index]}: {count} images.')
    
    return label_counting


# In[4]:


def show_images(dataset, class_names, num_images=16, rows=4, cols=4):
    """
    Adapted from: https://www.tensorflow.org/tutorials/load_data/images#visualize_the_data in order to 
    display a grid of 16 images from a dataset along with their corresponding class labels in a 4x4 grid.

    Input:
        dataset: tf.data.Dataset
            A TensorFlow dataset that yields image-label pairs.
        
        class_names: list
            A list of class names corresponding to the one-hot encoded labels. The index 
            of each class name in this list corresponds to the class label.
        
        num_images : int, optional
            The number of images to display. Default is 16.

        rows : int, optional
            The number of rows in the image grid. Default is 4.

        cols : int, optional
            The number of columns in the image grid. Default is 4.

    Output:
        None
            Simply displays the images and their corresponding labels in a grid layout.

    """
    plt.figure(figsize=(10, 10))
    for i, (imagem, label) in enumerate(dataset.take(num_images)):
        if i >= num_images:
            break
        plt.subplot(rows, cols, i+1)
        plt.imshow(imagem.numpy().astype("uint8"))
        plt.axis('off')
        class_name = class_names[label.numpy().argmax()]
        plt.title(f'{class_name}')
    plt.show()


# In[5]:


def plot_accuracy_f1_loss(history, epochs):
    """
    Adapted from: https://www.tensorflow.org/tutorials/images/classification to plot the
    training and validation accuracy, F1 score, and loss metrics during training.

    Input:
        history: keras.callbacks.History
            The history object returned by the model.fit() function.
        
        epochs: int
            The number of epochs used during training. This is used to determine the range for the x-axis in the plots.
    
    Output:
        None
            Display a grid of 3 plots: Training and Validation Accuracy, F1 Score and Loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    f1_score = history.history['f1_score']
    val_f1_score = history.history['val_f1_score']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(10,6))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, f1_score, label='Training F1 Score')
    plt.plot(epochs_range, val_f1_score, label='Validation F1 Score')
    plt.legend(loc='lower right')
    plt.title('F1 Score')

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.show()


# In[6]:


def save_model_log(model_name, history, epochs, file_path='log.json'):
    """ 
    Save the best performance metrics of the model to a JSON log file.

    Input:
        model_name: str
            The name of the model being logged.
            
        history: keras.callbacks.History
            The history object returned by the model.fit() function, containing the training metrics.

        epochs: int
            The number of epochs used during training. This is used to track the training process.

        file_path: str, optional (default='log.json')
            The file path where the log data will be saved. If the file exists, the data will be appended.

    Output:
        None
            The function saves the best metrics to a JSON file, without returning any value.
    """ 

    best_training_loss = min(history.history['loss'])
    best_training_accuracy = max(history.history['accuracy'])
    best_training_f1_score = max(history.history['f1_score'])

    best_val_loss = min(history.history['val_loss'])
    best_val_accuracy = max(history.history['val_accuracy'])
    best_val_f1_score = max(history.history['val_f1_score'])

    log_data = {
        'model_name': model_name,
        'epochs': epochs,
        'configurations': {
            'optimizer': 'Adam',
            'loss_function': 'CategoricalCrossentropy (from_logits=True)',
            'metrics': ['Accuracy', 'F1 Score']
        },
        'best_metrics': {
            'training': {
                'loss': best_training_loss,
                'accuracy': best_training_accuracy,
                'f1_score': best_training_f1_score
            },
            'validation': {
                'loss': best_val_loss,
                'accuracy': best_val_accuracy,
                'f1_score': best_val_f1_score
            }
        }
    }

    try:
        with open(file_path, 'a') as f:
            json.dump(log_data, f, indent=4)
            f.write('\n')
        
    except Exception as e:
        print(f'An error occurred: {e}')
        with open(file_path, 'w') as f:
            json.dump(log_data, f, indent=4)


# ## Load the dataset

# I have downloaded the dataset from the Kaggle repository available at the following link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data.
# 
# According to the dataset description, it contains approximately **25 000 images of size 150x150, distributed across 6 categories**: {0: Buildings, 1: Forest, 2: Glacier, 3: Mountain, 4: Sea, 5: Street}.
# 
# The data is already split into 3 subsets: training (~14 000 images), testing (~3 000 images), and prediction (~7 000 images).
# 
# Since the purpose of this project is to rapidly iterate through the training process, I will only upload one split, specifically the training set, which will be further divided later on.

# In[7]:


dataset_directory = '../Project_DeepLearning/input/seg_train'


# To properly load the dataset, I referred to the documentation available at https://keras.io/api/data_loading/image/#image-data-loading.
# 
# This will return a `tf.data.Dataset` object, where images are represented as tensors of shape `(batch_size, image_height, image_width, num_channels)` and labels are tensors of shape `(num_labels,)`, corresponding to one-hot encoded vectors.

# In[8]:


from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory

dataset = keras.utils.image_dataset_from_directory(
    dataset_directory,
    labels='inferred',                 # Labels will be named as the folders' names
    label_mode='categorical',          # To get the labels as one-hot encoded vectors
    batch_size=None,
    image_size=(150,150),              # Image Size as per the documentation of the dataset
    shuffle=True,
    seed=42
)


# In[9]:


dataset


# In[10]:


for image, label in dataset:
    print(f'Image shape: {image.shape}')
    print(f'Label shape: {label.shape}')
    print(label)
    break


# The shape of the image is (150, 150, 3), which means the images are in color with **3 RGB channels**.

# In[11]:


dataset.class_names


# ## Visualize the data

# In[12]:


labels_names = dataset.class_names

print(labels_names)


# In[13]:


show_images(dataset, labels_names)


# ## Explore the dataset

# Upon loading the dataset, we observe that it contains 14 034 images across 6 classes. However, will verify whether the classes are balanced or not.

# In[14]:


n_images_per_label = count_items_by_label(dataset, labels_names)


# In[15]:


# Visualize graphically the distribution of images per class

classes = [labels_names[i] for i in n_images_per_label.keys()]
counts = [n_images_per_label[i] for i in n_images_per_label.keys()]

plt.figure(figsize=(6, 6))
plt.barh(classes, counts, color='skyblue')
plt.xlabel('Number of images')
plt.title('Distribution of Images per Label')
plt.show()


# ## Transform the dataset

# ### Train, test, validation splits
# 
# Since the purpose of this project is to build a Neural Network from scratch and test different architectures, I will split the dataset into training, validation, and testing sets with fewer examples: 3 000 for training, 1 500 for validation, and 600 for testing.
# 
# For this, I have used the `split_dataset` function as referenced in the following link: https://keras.io/api/utils/python_utils/#splitdataset-function.

# In[16]:


from tensorflow.keras.utils import split_dataset

train_size = 3000
val_size = 1500
test_size = 600

# Dividing the dataset into the training dataset with 3000 images, and the rest

train_ds, rest = tf.keras.utils.split_dataset(
    dataset,
    left_size=train_size,
    seed=42
)

# Now with the rest dataset from above, I will divide into validation and test datasets

val_ds, test_ds = tf.keras.utils.split_dataset(
    rest,
    left_size=val_size,
    right_size=test_size,
    seed=42
)


# In[17]:


counting_iterator(train_ds, 'train'),
counting_iterator(val_ds, 'test'),
counting_iterator(test_ds, 'validation')


# In[18]:


label_n_train = count_items_by_label(train_ds, labels_names)

# Visualize graphically the distribution of images per class

classes_train = [labels_names[i] for i in label_n_train.keys()]
counts_train = [label_n_train[i] for i in label_n_train.keys()]

plt.figure(figsize=(6, 6))
plt.barh(classes_train, counts_train, color='skyblue')
plt.xlabel('Number of images')
plt.title('Distribution of Images per Label')
plt.show()


# In[19]:


label_n_val = count_items_by_label(val_ds, labels_names)


# In[20]:


label_n_test = count_items_by_label(test_ds, labels_names)


# ### Normalize Data

# In[21]:


def normalize_image(image, label):
    return tf.cast(image, tf.float32)/255.0, label


# In[22]:


normalize_train_ds = train_ds.map(normalize_image)
normalize_val_ds = val_ds.map(normalize_image)
normalize_test_ds = test_ds.map(normalize_image)


# In[23]:


# Checking if values were normalized: 

for image, label in normalize_train_ds.take(1):
    print(image.shape)
    print("Min:", tf.reduce_min(image).numpy())
    print("Max:", tf.reduce_max(image).numpy())


# ### Define Batch Size

# In[24]:


train_dataset = normalize_train_ds.batch(256)
test_dataset = normalize_test_ds.batch(256)
val_dataset = normalize_val_ds.batch(256)


# In[25]:


counting_iterator(train_dataset, 'train', 'batches')
counting_iterator(test_dataset, 'test', 'batches')
counting_iterator(val_dataset, 'validation','batches')


# The datasets have now been preprocessed and transformed, and are ready for the training and validation process.

# # Modelling

# In[26]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# ## Metrics

# I chose **Accuracy** and **F1 Score** as evaluation metrics. 
# 
# Accuracy provides a general sense of the model's performance, but it can be misleading in imbalanced datasets. Although the Intel Image Classification dataset is not imbalanced, I’ll use F1 Score to maintain robustness and prevent any potential issues in this regard. Additionally, the F1 Score harmonizes both precision and recall, ensuring a more comprehensive assessment of the model, rather than relying solely on accuracy.

# ## Baseline model

# I will start by building a model using only Dense layers, to establish a baseline for comparing the performance of Dense-based models with Convolutional-based ones.

# In[27]:


num_classes = len(labels_names)


baseline_model = Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes)
])


# In[28]:


baseline_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        tf.keras.metrics.F1Score(average='macro', name='f1_score')
    ]
)


# In[29]:


baseline_model.summary()


# In[30]:


epochs=50
history = baseline_model.fit(
  train_dataset,
  validation_data=val_dataset,
  epochs=epochs
)


# In[31]:


save_model_log('baseline_model', history, epochs)


# In[32]:


plot_accuracy_f1_loss(history, epochs)


# It is now evident why Dense layers are not ideal for extracting meaningful features from images. Not only are the metrics generally underwhelming, but the validation metrics also quickly stagnated around 50%, indicating poor performance.

# ## Models with Convolutional Layers
# 
# I will now build a neural network using Convolutional layers, which are known to be more effective for image classification. Therefore, I expect it to outperform the baseline model.
# 
# I’ll start with a shallow architecture and gradually increase its depth, exploring techniques such as Padding, Max Pooling and regularization techniques along the way.

# In[34]:


num_classes = len(labels_names)

cnn_base = Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes)
])


# In[35]:


cnn_base.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        tf.keras.metrics.F1Score(average='macro', name='f1_score')
    ]
)


# In[36]:


cnn_base.summary()


# In[37]:


epochs = 20

history_cnn_base = cnn_base.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)


# In[38]:


save_model_log('cnn_base', history_cnn_base, epochs)


# In[39]:


plot_accuracy_f1_loss(history_cnn_base, epochs)


# In[40]:


print(f'The best Accuracy achieved in training with cnn_base was {round(max(history_cnn_base.history['accuracy']), 4)}.')
print(f'The best F1 Score achieved in training with the cnn_base was {round(max(history_cnn_base.history['f1_score']), 4)}.')


# The metrics from the `cnn_base` suggest that the training dataset has been memorized, as we observe nearly the maximum levels of accuracy and F1 score on the training set, which do not translate to the validation set, indicating its inefficiency in generalizing.
# 
# One potential cause of this overfitting could be the number of parameters, as the model contains around 22 million parameters according to the `cnn_base.summary()` function. 
# 
# Considering the size of the training dataset, this may be excessive. Therefore, I will now build a model incorporating the concept of **Max Pooling**, as well as **Padding**, while also increasing the depth of the architecture.

# In[41]:


num_classes = len(labels_names)

cnn_v2 = Sequential([
    layers.Input(shape=(150,150,3)),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes)
])


# In[42]:


cnn_v2.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        tf.keras.metrics.F1Score(average='macro', name='f1_score')
    ]
)


# In[43]:


cnn_v2.summary()


# Using the `cnn_v2.summary()` function, we can clearly see the difference in the number of parameters when applying Max Pooling versus not. The model went from 22 million parameters down to just 1.3 million, while increasing its depth.

# In[44]:


epochs = 20

history_cnn_v2 = cnn_v2.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)


# In[45]:


save_model_log('cnn_v2', history_cnn_v2, epochs)


# In[46]:


plot_accuracy_f1_loss(history_cnn_v2, epochs)


# In[47]:


print(f'The best Accuracy achieved with cnn_v2 was {round(max(history_cnn_v2.history['accuracy']), 4)}.')
print(f'The best F1 Score achieved with the cnn_v2 was {round(max(history_cnn_v2.history['f1_score']), 4)}.')


# The divergence in metrics between the training and validation datasets suggests evidence of overfitting. However, since the validation metrics are still showing an upward trend, I will train the model for a few more epochs.

# In[48]:


epochs = 10

history_cnn_v2 = cnn_v2.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)


# In[49]:


plot_accuracy_f1_loss(history_cnn_v2, epochs)


# With both CNN models, `cnn_base` and `cnn_v2`, we can now observe the impact of using Max-Pooling. We transitioned from a 21M-parameter model that instantly overfitted the training data and performed poorly on the validation set, to a deeper model with only 1.5M parameters that showed slight improvements in the validation metrics — logs are available in the log.json file.
# 
# However, these improvements were modest (we increased accuracy from 68% to 72%), and overfitting continues to be an issue for both models.
# 
# Therefore, I will incorporate **regularization** and **dropout** techniques into the `cnn_v2` architecture to assess whether they can positively impact the model's generalization and help prevent overfitting.

# In[50]:


from tensorflow.keras import regularizers

num_classes = len(labels_names)

cnn_reg = Sequential([
    layers.Input(shape=(150,150,3)),

    layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),

    layers.Flatten(),                        
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(num_classes)
])


# In[51]:


cnn_reg.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        tf.keras.metrics.F1Score(average='macro', name='f1_score')
    ]
)


# In[52]:


cnn_reg.summary()


# In[53]:


epochs = 25

history_cnn_reg = cnn_reg.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)


# In[54]:


plot_accuracy_f1_loss(history_cnn_reg, epochs)


# In[55]:


epochs = 15

history_cnn_reg = cnn_reg.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)


# In[56]:


save_model_log('cnn_reg',history_cnn_reg,epochs)


# In[57]:


plot_accuracy_f1_loss(history_cnn_reg, epochs)


# By incorporating regularization and dropout, we observed that the model took longer to reach the point of overfitting, and the validation metrics showed a slight improvement (increasing from 72% accuracy to 76%), as seen in the log.json file.
# 
# Nevertheless, I still aim to achieve higher metrics on the validation set. Therefore, I will apply transfer learning, leveraging pre-trained models to boost accuracy and F1 score, and further improve generalization on unseen data.

# ## Transfer Learning
# 
# The list of the available models can be checked at the follow link: https://keras.io/api/applications/#available-models.
# 
# Given the size of our dataset and the goal of iterating quickly through the training process, I will use the **MobileNet** architecture, as it is one of the lightest in terms of memory usage and number of parameters, which I believe is well-suited for our problem.

# In[ ]:


from tensorflow.keras.applications import MobileNet


# In[59]:


MobileNet = MobileNet(
    input_shape=(150,150,3),
    include_top=False,
    weights='imagenet',
)

MobileNet.summary()


# As I intend to use the pre-trained and already optimized weights, I will freeze these parameters.

# In[60]:


for layer in MobileNet.layers:
    layer.trainable = False


# In[84]:


# To check that there are no Treinable params now:

MobileNet.summary()


# In[61]:


num_classes = len(labels_names)

cnn_mobilenet = Sequential([
    MobileNet,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes)
])


# In[62]:


cnn_mobilenet.summary()


# In[63]:


cnn_mobilenet.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        tf.keras.metrics.F1Score(average='macro', name='f1_score')
    ]
)


# Since I expect this to be the best-performing model among all those created so far, I will save its structure (both architecture and weights) from the best-performing epoch — specifically, the one that achieved the highest accuracy on the validation set.
# 
# To do this, I will use the `ModelCheckpoint()` class, as described at the following link: https://keras.io/api/callbacks/model_checkpoint/ , and incorporate it into the `.fit()` method.

# In[64]:


from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='cnn_mobilenet.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False
)


# In[65]:


epochs = 15

history_mobilenet = cnn_mobilenet.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[checkpoint]
)


# In[66]:


save_model_log('cnn_mobilenet',history_mobilenet,epochs)


# In[67]:


plot_accuracy_f1_loss(history_mobilenet, epochs)


# Although the training metrics are still on an upwards trend, in the last few epochs the training and validation metrics have started to diverge, so I’ve decided not to continue training this architecture.
# 
# Even so, this clearly demonstrates the strength of using transfer learning instead of building a neural network from scratch, as the validation metrics were already better, even before any training, than those of all the previously tested models.
# 
# That said, since this is our best-performing model so far, I will use the saved iteration to evaluate its performance on the test dataset.

# ## Assessing the best model on the test dataset

# Now that the best-performing iteration based on validation accuracy has been saved to the filepath **cnn_mobilenet.keras**, I will load this model using the `load_model()` class and evaluate its performance on the test dataset.

# In[68]:


from tensorflow.keras.models import load_model

best_cnn = load_model('cnn_mobilenet.keras')

test_loss, test_accuracy, test_f1_score = best_cnn.evaluate(test_dataset)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}%")
print(f"Test F1 Score: {test_f1_score:.4f}")


# Since the training methodology has been validated and the model has achieved strong performance across the training, validation, and test datasets, I will now proceed to retrain this architecture using the full dataset in order to maximize the data available for final model training.

# # Final Training

# We still have the entire dataset stored in the **dataset** variable, which has already been shuffled. However, I will need to normalize the data, split it into batches, and apply `prefetch(tf.data.AUTOTUNE)`: This is a best practice in data pipelines, especially when training with large datasets, as it helps optimize performance by overlapping data loading and model training.

# In[70]:


len(dataset)


# In[75]:


dataset = dataset.map(normalize_image)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


# In[81]:


save_best = ModelCheckpoint(
    filepath='final_cnn.keras',
    monitor='accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False
)


# In[82]:


best_cnn.fit(
    dataset,
    epochs=10,
    callbacks=[save_best]
)


# The model was trained above for a total of 30 epochs (10+10+10).
# 
# During the last 10 epochs, all metrics began to exhibit a bit of a ping-pong effect, indicating that the model was struggling to improve consistently. As a result, I decided to stop the training.
# 
# The best iteration in terms of accuracy was saved in the file `final_cnn.keras`, achieving **94% accuracy**. This represents the final performance for the problem, and the model would now be ready to be evaluated on new datasets and further improved.
