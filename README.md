# PneumoSense
Chest X-Ray Pneumonia Detection:
This notebook implements a Convolutional Neural Network (CNN) to classify Chest X-ray images into 'Pneumonia' or 'Normal' categories. The project covers data loading, preprocessing, model training and performance analysis.

1). Setup and Data Loading
This section initializes the environment by importing necessary libraries and loading the dataset.
The dataset: 'chest-xray-pneumonia', is downloaded from KaggleHub and prepared for processing.

1.1) Import Libraries
All essential Python libraries for data manipulation (numpy, pandas), visualization (matplotlib, seaborn), deep learning (Keras, TensorFlow), and image processing (cv2) are imported. This ensures all required functionalities are available for subsequent steps.

1.2) Load Dataset
The X-ray images are loaded from the Kaggle dataset.
Get_training_data is used to read image files, resize them to 150x150 pixels, convert them to grayscale (standard for X-ray images), and categorize them as 'PNEUMONIA' (class 0) or 'NORMAL' (class 1). The data is then split into training, testing, and validation sets.

2). Data Visualization & Preprocessing
This section focuses on understanding the data distribution, preparing images for the neural network, and applying data augmentation techniques to prevent overfitting.

2.1) Dataset Analysis
The distribution of 'Pneumonia' and 'Normal' cases in the training dataset is analyzed to understand class balance.
Visual examples of both Pneumonia and Normal X-ray images are displayed to provide a visual understanding of the data.

2.2) Data Splitting and Reshaping
The raw image data is separated into features (x) and labels (y) for training, validation, and testing.
To manage computational resources and allow for faster experimentation, the dataset size is reduced by sampling a subset of the original data using train_test_split with stratified sampling to maintain class balance. Images are then reshaped to (img_size, img_size, 1) to fit the input requirements of a CNN, and labels are converted to NumPy arrays.

2.3) Class Weight Calculation
Class weights are computed to address potential class imbalance in the dataset.
This helps the model pay more attention to the minority class during training, preventing bias towards the majority class.

2.4) Mixed Precision Training
Mixed precision training is enabled to leverage float16 for most computations while keeping numerically stable operations in float32. This reduces memory usage and speeds up training, potentially reducing the problem of overfitting.

2.5) Data Normalization
Image pixel values, initially in the range 0-255, are normalized to 0-1 by dividing by 255.
This step is crucial for optimal performance of neural networks.

2.6) Data Augmentation
To prevent overfitting and improve the model's generalization capabilities, data augmentation techniques are applied.
These include random rotations, zooms, shifts, horizontal flips, brightness adjustments, and contrast changes.
Two methods are used:

ImageDataGenerator (Legacy): Configures a Keras ImageDataGenerator for various transformations.

tf.data pipelines (Recommended): Utilizes tf.data.Dataset with Keras preprocessing layers (RandomRotation, RandomZoom, RandomTranslation, RandomFlip, RandomBrightness, RandomContrast) for more efficient and flexible augmentation.

3) Training the Model
This section defines the CNN architecture, compiles the model, and initiates the training process.

3.1) Model Architecture
A sequential CNN model is constructed using tf.keras.Sequential.
It consists of multiple convolutional (Conv2D) and max-pooling (MaxPool2D) layers for feature extraction, interleaved with BatchNormalization layers for stable training.
A Flatten layer converts the 3D feature maps into a 1D vector, followed by dense layers (Dense) for classification.
A sigmoid activation function is used in the final output layer for binary classification.

Conv2D: Applies filters to detect features.
BatchNormalization: Normalizes activations, speeding up training and improving stability.
MaxPool2D: Reduces spatial dimensions, decreasing computation and helping prevent overfitting.
Flatten: Transforms 3D output to 1D for dense layers.
Dense: Fully connected layers for classification.
Sigmoid: Output activation for binary classification, yielding probabilities between 0 and 1.
3.2 Model Compilation
The model is compiled with the rmsprop optimizer, binary_crossentropy loss function (suitable for binary classification), and accuracy as the evaluation metric.

3.3) Learning Rate Reduction
A ReduceLROnPlateau callback is used to dynamically adjust the learning rate during training.
If the validation accuracy does not improve for a certain number of epochs (patience), the learning rate is reduced by a factor (factor).

3.4) Model Training
The model is trained using the augmented training data (datagen.flow(x_train, y_train)) and validated with datagen.flow(x_val, y_val) over a specified number of epochs (e.g., 12).
The learning_rate_reduction callback is applied during training.

4) Analysis After Model Training
After the model is trained, its performance is evaluated using various metrics and visualizations.

4.1) Model Evaluation
The model's loss and accuracy are evaluated on the unseen test dataset (test_ds) to determine its generalization performance.

4.2) Training and Validation Plots
Plots of training and validation accuracy and loss over epochs are generated.
These plots help visualize the learning process and identify potential issues like overfitting or underfitting.

4.3) Predictions and Classification Report
Predictions are made on the test set, converting the model's probability outputs to binary class labels (0 or 1).
A classification_report is generated specifically for the 'Pneumonia' class (class 0), providing detailed metrics such as precision, recall, and F1-score.

4.4) Confusion Matrix
A confusion matrix is calculated and visualized as a heatmap.
This matrix provides a comprehensive summary of the model's classification performance, showing true positives, true negatives, false positives, and false negatives.
The matrix is converted to a Pandas DataFrame for clearer indexing with '0' and '1' representing the classes and then displayed with 'Pneumonia' and 'Normal' labels.

4.5) Correct and Incorrect Predictions
The indices of correctly and incorrectly classified images in the test set are identified by comparing the model's predictions with the actual labels.
Examples of both correctly and incorrectly classified images (up to 6 of each) are then displayed, showing the predicted and actual class for visual inspection.
