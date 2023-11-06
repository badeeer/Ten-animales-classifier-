# Ten-animales-classifier-

The Python script is a comprehensive image classification project that operates as follows:

1. **Data Organization**: It starts by specifying input and output directories for the dataset and then organizes the data into subdirectories for training, validation, and test sets. Class folders are split into these sets based on a specified ratio, and a random selection of images from the training set is displayed.

2. **Data Loading**: It loads the dataset using TensorFlow's `image_dataset_from_directory`. The dataset is split into training, validation, and test datasets with batch size and image size defined.

3. **CNN Model Building**: A Convolutional Neural Network (CNN) model is defined for image classification. This model includes convolutional layers, max-pooling layers, and a dense output layer with softmax activation. The model is configured for training, specifying the loss function, optimizer, and metrics.

4. **Model Training**: The model is trained on the training dataset with early stopping and model checkpoint callbacks to monitor and save training progress. The training history is stored.

5. **Feature Extraction**: The script leverages a pre-trained VGG16 model to extract features from the images. A new model for feature extraction and training is defined. This model includes a dense output layer with softmax activation. This new model is configured for training, with the same early stopping and model checkpoint callbacks.

6. **Feature Model Training**: The new model is trained on the extracted features from the training and validation datasets. The training history is stored.

7. **Fine-Tuning**: The pre-trained VGG16 model is fine-tuned by allowing certain layers to be trainable while freezing others. The script configures and trains the fine-tuned model using the feature data from the training and validation datasets.

8. **Model Evaluation**: Finally, the script evaluates the performance of the fine-tuned model on the test dataset, providing the test accuracy.

Overall, this code showcases a complete workflow for image classification, including data organization, model building, training, feature extraction, and fine-tuning, all implemented using TensorFlow and a pre-trained VGG16 model.
