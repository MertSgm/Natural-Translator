
# Natural Translator
Natural_Translator is a machine learning project that can be used to translate a source language to a target language.

## Installation
Install [Python 3.8](https://www.python.org/downloads/release/python-380/).

Install [TensorFlow](https://www.tensorflow.org/install).

To make use of an NVIDIA GPU during model training,
[CUDA](https://developer.nvidia.com/accelerated-computing-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) are required.

## Usage
To preface, the project makes use of two models for natural language translation. <br>
The first is the [Feed Forward Neural Network](res/models/ffnn/model_figures/FFNN_TEST_MODEL.png), followed
by the [Recurrent Neural Network](res/models/rnn/model_figures/RNN_TEST_MODEL.png). <br>
This project also features the Self Attention Neural Network [[1]](#1).
It can be seen for example purposes, but is not functional.

To use the project, simply follow the instructions in [main.py](main.py).


## Authors
[MertSgm](https://github.com/MertSgm) and [Robino-CK](https://github.com/Robino-CK)

## References
<a id=1>[1]</a>
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser, I. Polosukhin <br>
Attention is all you need.<br>
In Advances in Neural Information Processing Systems, pp. 6000–6010, 2017.

## License
MIT