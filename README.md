# How we begun
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xRaw1bmtqAueGvcBBVhPxr_laKkD3yjs?usp=sharing)

* It all started with a fork of https://github.com/k7922n/Seq2seq-Chatbot-With-Deep-Reinforcement-Learning. Which is a wonderful repo that is not maintained well.
* The repo is not yet updated for 2 years and it was difficult to work with the old code. Since now people uses TensorFlow 2.x it was challenging to debug and change the code.
* We choose TensorFlow 1.x. But the core_rnn library was changed to rnn there is no core_rnn_cell_impl library, so we modified the code. There are various dependencies problem happened througout the program.

* We finished modification and moved towards the integration of PyTorch model which was trained on Stanford Question Answering Dataset (SQuAD) and various COVID-19 articles. This is fast and gives rapid response to the user. 
# How it works

* We upload PDF files of various articles, news related to the desired goal in the backend, the PyTorch model will do a semantic search and find the best answer. The question and answers are saved as source and target file to train our Seq2Seq model. The model gets fine tuned with pretrained sentimental anlalysis model using RL which will be deployed in the edge.
 # RUN on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xRaw1bmtqAueGvcBBVhPxr_laKkD3yjs?usp=sharing)
