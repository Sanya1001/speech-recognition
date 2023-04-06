# speech-recognition
This branch explores adding attention layers to previous architecture to see if there is a difference in them.

For details to architecture, see `BiLSTM with Attention.txt` and `BiLSTM with Attention 2.txt`

Conclusion: They are pretty similar, though rate of decrease of loss might be faster with MultiHead attention.

Problems:
Training can only be tested with 1000 examples at most. The model is not learning well, and on checking some intermediate outputs often predicts blank tokens of the expected sequence length. Possible causes: lack of proper training since audio data is complex to learn, and we only tested using 1000 examples.

However, the model runs without errors. Evaluation using `editdistance` is in place.

Future Direction:
- Look into details of loss function
- Train this model on a larger sample
