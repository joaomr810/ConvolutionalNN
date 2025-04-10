# Convolutional Neural Network (CNN) for Image Classification

This repository contains the files where neural networks were built using **TensorFlow** and **Keras** to achieve the best possible metrics for the problem of **Image Classification** using the **Intel Image Classification** dataset. After testing various models, architectures, and techniques, including the application of **transfer learning**, the final model achieved a **94% accuracy**.

## Contents

| Folder/File Name    | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| input/seg_train     | Folder containing the Intel Image Classification dataset used for training and evaluation |
| Project_CNN.ipynb   | Jupyter Notebook where the full project was developed, including data exploration, model training, evaluation, and conclusions |
| Project_CNN.py      | Python script version of the notebook above |
| final_cnn.keras     | Saved model file containing the best-trained Convolutional Neural Network (CNN) with a 94% accuracy on the full dataset.    |
| cnn_mobilenet.keras | Saved model instance that achieved the best metrics across the training, validation, and testing datasets when using the smaller training dataset |
| log.json            | JSON file logging the best model instances during training, storing metrics for both training and validation  |
| requirements.txt    | File listing all Python dependencies required to run the project  |


## Dependencies

This project was developed using **Python** and the **TensorFlow** framework, along with additional libraries such as **NumPy**, **Pandas**, and **Matplotlib**.

To install all the required packages, you can use the `requirements.txt` file:

`pip install -r requirements.txt`

Alternatively, the main dependencies can be installed manually with the following command:

`pip install tensorflow numpy pandas matplotlib`

## Overview of the notebook

In this notebook, I imported the **Intel Image Classification** training dataset, which contains around 14 000 images distributed across 6 classes: buildings, forest, glacier, mountain, sea, and street.

The goal of the project was to build neural networks from scratch and iterate quickly through various training processes and architectures to achieve strong metrics **(>90% accuracy / >0.9 F1 Score)**. Therefore, the dataset was initially split into smaller subsets for training, validation, and testing.

After an **exploratory data analysis**, where the images were normalized, the data splits were ready for training.

I began with a simple neural network using only dense layers, to serve as a baseline before introducing **convolutional layers**, which are much more effective for image classification tasks.

Throughout the different architectures, I explored concepts like **Max Pooling** and **Padding**, along with overfitting prevention techniques such as **regularization** and **dropout**.

Finally, I used **Transfer Learning** with the pretrained **MobileNet** model and customized the final layers for this classification task. This model achieved the best performance on the training and validation datasets (91% accuracy / 0.91 F1 Score). After saving the architecture and weights, it was tested on the test dataset and achieved strong results as well.

With the training methodology validated, I proceeded to train the final model using the entire dataset to maximize the use of all available data. The resulting model achieved **94% accuracy** and is saved in the `final_cnn.keras` file, ready to be used for image classification tasks on new datasets and be further improved.

## Possible Use Cases

- **Satellite or Drone Image Classification**: Automatically classify aerial images into land types
- **Image Organization for Visual Platforms**: Automatically tag and group large image datasets by landscape type
- **Environmental Monitoring Systems**: Detect and track changes in land use or vegetation cover over time
- **Urban Planning Support**: Categorize urban vs. natural areas to assist in infrastructure planning
- **Tourism & Geo-Recommendation Apps**: Recommend destinations based on the user's preferred landscape type (e.g., mountains, sea, forest) identified from uploaded photos

## Future Improvements

- **Hyperparameter Tuning**: Fine-tuning of model parameters, such as learning rate, batch size, dropout and regularization rate, number of layers, kernel sizes, and other hyperparameters, to potentially improve performance.
- **Advanced Architectures**: Experimenting with more advanced CNN architectures (e.g., ResNet, EfficientNet, InceptionV3) to explore their techniques and determine if implementing any of them can lead to even higher performance.
- **Model Evaluation on New Datasets**: Testing the final model on new datasets to assess its performance across various image classification tasks. This will help in identifying areas for further improvement and adapting the model to new data. Regular updates and fine-tuning of the model based on these evaluations will ensure the model is always optimized and ready for deployment.