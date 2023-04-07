# speech-recognition

This branch trains and tests a Reccurent Neural Network on a section of the audio dataset. RNNs are designed to recognize the sequential characteristics of data and thereafter using the patterns to predict the coming scenario. For most speech recognition problems, it could be a decent model to use. For our data, however, there are sequences that are very long, and computation is slow with RNN. Vanishing gradients may persist, which is why expected accuracy with this model is low and may not be our top choice for the given scenario.

Currently, there are some runtime errors in the training program related to varying dimensions of spectrograms. Iterating through a dataloader object needs spectrogram tensors to be of equal size, which is not the case for our dataset. Our next objective is to truncate or pad the spectrograms in a way that fixes the error as well as retain useful information about the spectrograms for training purposes. Once that is done, further analysis of the performance of the model on train and test data will be provided.

```
RNeuralNetwork(
  (RNN): RNN(1, 1, num_layers=3, dropout=0.01)
)
```
