# Speech Emotion Recognition for Customer Satisfaction

This project aims to develop a system capable of recognizing and analyzing emotions in customer speech to gauge satisfaction levels more accurately and enhance the overall customer service experience.

## Dataset
The dataset used for this project was obtained from [Kaggle](https://www.kaggle.com). It contained 4 most popular datasets in English: Crema, Ravdess, Savee and Tess. Each of them contains audio in .wav format with some main labels.

## Tools and Technologies
- Programming Languages: Python
- Libraries:librosa, numpy, matplotlib, scikit-learn, tensorflow (or keras)
- Tools: Jupyter notebooks, Git
- Visualization: Matplotlib, Seaborn, Plotly

## Usage
1. Data Preprocessing:

-Convert raw audio files to log-mel spectrograms and average the values.
-Extract features and labels for training and testing.
-Perform data augmentation by adding Gaussian noise, applying time stretching, and increasing speed and pitch.

2. Model Training:

Train the 1D CNN model using the preprocessed data.

3. Model Evaluation:

Evaluate the trained model on the test dataset and generate a classification report.

4. Visualization:

Plot the training and validation accuracy and loss.

## Results
-Initial model evaluation with a dummy classifier achieved 11.46% accuracy.
-Decision tree classifier improved the accuracy to 31.25%.
-The 1D CNN model trained for 20 epochs achieved an accuracy of 11.85% and a loss of 2.0676.
-With 200 epochs, the 1D CNN model achieved a test accuracy of 51.74% and a loss of 1.4048.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue.
