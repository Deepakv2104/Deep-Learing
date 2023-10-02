# Recurrent Models Comparison on IMDB Sentiment Analysis

## Overview
This code aims to compare the performance of three popular recurrent neural network (RNN) architectures—SimpleRNN, GRU, and LSTM—on the task of sentiment analysis using the IMDB dataset.

## Dataset
The IMDB dataset contains 50,000 movie reviews (25,000 training and 25,000 test) labeled as positive (1) or negative (0). The dataset is loaded from Keras's datasets module.

## Preprocessing
1. The dataset is loaded with a vocabulary size of 10,000 (`max_words`), which means only the top 10,000 most frequent words are considered.
2. Reviews are padded or truncated to a fixed length of 250 words (`max_length`) to ensure consistency in input dimensions.

## Model Architectures
Three models with the following architectures are constructed:
1. **SimpleRNN Model**: An embedding layer followed by a SimpleRNN layer and a dense output layer.
2. **GRU Model**: An embedding layer followed by a GRU layer and a dense output layer.
3. **LSTM Model**: An embedding layer followed by an LSTM layer and a dense output layer.

All models use:
- An embedding dimension of 64.
- A batch size of 128 for training.
- Binary cross-entropy loss, given the binary classification task.
- Adam optimizer.
- 3 training epochs.

## Training and Evaluation
Each model is trained on the training dataset with a validation split of 20%. After training, each model is evaluated on the test dataset to calculate its accuracy. The training accuracy of each model over the epochs is then plotted for comparison.

## Visualization
The training accuracies of the three models are plotted against the number of epochs. This allows for a visual comparison of how each model's accuracy evolves over time during training.

## Required Libraries
- numpy
- matplotlib
- tensorflow

## How to Run
1. Ensure that you have all the required libraries installed.
2. Copy and paste the code into a Python environment or script.
3. Run the code. The models will be trained, evaluated, and the training accuracy will be plotted.

## Note
Training deep learning models can be computationally intensive and time-consuming. Ensure that you have a suitable environment (preferably with a GPU) for training.

## Future Enhancements
1. Experiment with different hyperparameters such as embedding dimensions, RNN units, and learning rates.
2. Incorporate more advanced techniques like dropout or batch normalization for better generalization.
3. Test with other recurrent architectures or model configurations.
4. Add functionality to save the trained models for future use or deployment.
