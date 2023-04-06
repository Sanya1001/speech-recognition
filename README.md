# speech-recognition

## Results so far
We have attempted to train a variety of models (and apply a few pre-trained models), with varying levels of success.

We have tried the following models, each in their own branch:
1. [SVM](https://github.com/Sanya1001/speech-recognition/tree/SVM)
2. [CNN](https://github.com/Sanya1001/speech-recognition/tree/cnn)
2. [BiLSTM](https://github.com/Sanya1001/speech-recognition/tree/attention)
3. [BiGRU](https://github.com/Sanya1001/speech-recognition/tree/bilstm) (on the confusingly-named `bilstm` branch)
4. [RNN](https://github.com/Sanya1001/speech-recognition/tree/standard-rnn)
4. [OpenAI Whisper](https://github.com/Sanya1001/speech-recognition/tree/whisper) (pre-trained)

We have been training our models on the [LibriTTS](https://pytorch.org/audio/stable/generated/torchaudio.datasets.LIBRITTS.html#torchaudio.datasets.LIBRITTS) dataset, which contains ~585 hours of English text read aloud, captured at a 24kHz sampling rate. Each training example is a recording of a single sentence.

Training on the entire dataset has been infeasible for us, so our models are trained on a subset of the data.

### SVM
The SVM model applies a Support Vector Machine to the problem of audio classification. It uses the audio waveforms as input and outputs a single label for each audio file. The label is based on the file name, and represents which class the audio file belongs to.

### CNN
Like the SVM model, the CNN applies a Convolutional Neural Network to the problem of audio classification. It also uses the audio waveforms as input and outputs a single label for each audio file.

The CNN model has the following structure:
```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (conv_relu_stack): Sequential(
    (0): Conv1d(8, 13, kernel_size=(3,), stride=(1,))
    (1): ReLU()
    (2): MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.3, inplace=False)
    (4): Conv1d(16, 11, kernel_size=(3,), stride=(1,))
    (5): ReLU()
    (6): MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (7): Dropout(p=0.3, inplace=False)
    (8): Conv1d(32, 9, kernel_size=(3,), stride=(1,))
    (9): ReLU()
    (10): MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (11): Dropout(p=0.3, inplace=False)
    (12): Conv1d(64, 7, kernel_size=(3,), stride=(1,))
    (13): ReLU()
    (14): MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (15): Dropout(p=0.3, inplace=False)
    (16): Linear(in_features=256, out_features=256, bias=True)
    (17): ReLU()
    (18): Dropout(p=0.3, inplace=False)
    (19): Linear(in_features=128, out_features=128, bias=True)
    (20): ReLU()
    (21): Dropout(p=0.3, inplace=False)
    (22): Linear(in_features=10, out_features=10, bias=True)
    (23): Softmax(dim=None)
  )
)
```

### BiLSTM
The BiLSTM model has the following structure:
```
SpeechRecognitionModel(
  (cnn): Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (cnn_layers): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (fully_connected): Linear(in_features=2048, out_features=512, bias=True)
  (birnn_layers): Sequential(
    (0): BidirectionalLSTM(
      (BiLSTM): LSTM(512, 512, batch_first=True, bidirectional=True)
      (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (1): BidirectionalLSTM(
      (BiLSTM): LSTM(1024, 512, bidirectional=True)
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (2): BidirectionalLSTM(
      (BiLSTM): LSTM(1024, 512, bidirectional=True)
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (3): BidirectionalLSTM(
      (BiLSTM): LSTM(1024, 512, bidirectional=True)
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (4): BidirectionalLSTM(
      (BiLSTM): LSTM(1024, 512, bidirectional=True)
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (classifier): Sequential(
    (0): Linear(in_features=1024, out_features=512, bias=True)
    (1): GELU(approximate='none')
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=512, out_features=29, bias=True)
  )
  (attention): MultiHeadAttention(
    (q_linear): Linear(in_features=512, out_features=512, bias=True)
    (v_linear): Linear(in_features=512, out_features=512, bias=True)
    (k_linear): Linear(in_features=512, out_features=512, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_linear): Linear(in_features=512, out_features=512, bias=True)
  )
)
```

This model uses CTC Loss to train, with an Adam optimizer. It is able to train successfully on the LibriTTS dataset, although on our machines it is challenging to train/test on more than about 1000 examples at most. The model is not learning well, and on checking some intermediate outputs often predicts blank tokens of the expected sequence length. Possible causes: lack of proper training since audio data is complex to learn, and we only tested using 1000 examples.

Testing results:  
Average character edit distance: 98.6  
Average word edit distance: 18.2

### BiGRU
The BiGRU model has a similar structure to the BiLSTM, but with a few modifications (primarily the use of a GRU rather than LSTM):
```
SpeechRecognitionModel(
  (cnn): Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (rescnn_layers): Sequential(
    (0): ResidualCNN(
      (cnn1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (cnn2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (dropout1): Dropout(p=0.1, inplace=False)
      (dropout2): Dropout(p=0.1, inplace=False)
      (layer_norm1): CNNLayerNorm(
        (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      )
      (layer_norm2): CNNLayerNorm(
        (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      )
    )
    (1): ResidualCNN(
      (cnn1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (cnn2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (dropout1): Dropout(p=0.1, inplace=False)
      (dropout2): Dropout(p=0.1, inplace=False)
      (layer_norm1): CNNLayerNorm(
        (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      )
      (layer_norm2): CNNLayerNorm(
        (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      )
    )
    (2): ResidualCNN(
      (cnn1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (cnn2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (dropout1): Dropout(p=0.1, inplace=False)
      (dropout2): Dropout(p=0.1, inplace=False)
      (layer_norm1): CNNLayerNorm(
        (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      )
      (layer_norm2): CNNLayerNorm(
        (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (fully_connected): Linear(in_features=2048, out_features=512, bias=True)
  (birnn_layers): Sequential(
    (0): BidirectionalGRU(
      (BiGRU): GRU(512, 512, batch_first=True, bidirectional=True)
      (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (1): BidirectionalGRU(
      (BiGRU): GRU(1024, 512, bidirectional=True)
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (2): BidirectionalGRU(
      (BiGRU): GRU(1024, 512, bidirectional=True)
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (3): BidirectionalGRU(
      (BiGRU): GRU(1024, 512, bidirectional=True)
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (4): BidirectionalGRU(
      (BiGRU): GRU(1024, 512, bidirectional=True)
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (classifier): Sequential(
    (0): Linear(in_features=1024, out_features=512, bias=True)
    (1): GELU(approximate='none')
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=512, out_features=29, bias=True)
  )
)
```

Like the BiLSTM, this model uses CTC Loss and an Adam optimizer. It is also able to train on the LibriTTS dataset, but again, our computers run out of memory when attempting to train on more than a few hundred examples. This model also produces many blank tokens as output, and is not learning well. Additionally, it sometimes starts to give NaN outputs after a few epochs of training.

Testing results:  
Average character edit distance: 105.6  
Average word edit distance: 19.9

### RNN
This branch trains and tests a Reccurent Neural Network on a section of the audio dataset. RNNs are designed to recognize the sequential characteristics of data and thereafter using the patterns to predict the coming scenario. For most speech recognition problems, it could be a decent model to use. For our data, however, there are sequences that are very long, and computation is slow with RNN. Vanishing gradients may persist, which is why expected accuracy with this model is low and may not be our top choice for the given scenario.

Currently, there are some runtime errors in the training program related to varying dimensions of spectrograms. Iterating through a dataloader object needs spectrogram tensors to be of equal size, which is not the case for our dataset. Our next objective is to truncate or pad the spectrograms in a way that fixes the error as well as retain useful information about the spectrograms for training purposes. Once that is done, further analysis of the performance of the model on train and test data will be provided.

### OpenAI Whisper (pre-trained)
We tested the OpenAI Whisper model on the full LibriTTS testing dataset (4837 examples) and found the following results:  
Average character edit distance: 5.7  
Average word edit distance: 4.0  

Pretty impressive!
