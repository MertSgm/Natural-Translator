"""
The config.py file has all necessary inputs for the functions listed in main.py.
Simply change the values here in order to test new configurations.
"""

from src.preprocessing.dictionary import load_dictionary

# -- FOR EXAMPLE PURPOSES --
# data paths
example_train_source = "res/data/train/example_train_test_file_de"
example_train_target = "res/data/train/example_train_test_file_en"
example_evaluate_source = "res/data/evaluate/example_evaluate_test_file_de"
example_evaluate_target = "res/data/evaluate/example_evaluate_test_file_en"

# bpe
example_bpe_model_name = "EXAMPLE_BPE_MODEL"
example_bpe_model_path = f"res/bpe_models//{example_bpe_model_name}"
example_bpe_encode_output = f"{example_evaluate_target}_encoded"
example_bpe_decode_output = f"{example_bpe_encode_output}_decoded"
example_bpe_readable_output_target = f"{example_bpe_model_path}_readable"
# --------------------------

# data paths:
# evaluate
data_path_evaluate_source = "res/data/evaluate/multi30k.dev.de"
data_path_evaluate_target = "res/data/evaluate/multi30k.dev.en"
data_path_evaluate_bpe_source = "res/data/evaluate/multi30k.bpe.dev.de"
data_path_evaluate_bpe_target = "res/data/evaluate/multi30k.bpe.dev.en"

# train
data_path_train_source = "res/data/train/multi30k.de"
data_path_train_target = "res/data/train/multi30k.en"
data_path_train_bpe_source = "res/data/train/multi30k.bpe.de"
data_path_train_bpe_target = "res/data/train/multi30k.bpe.en"

# bpe:
bpe_model_name = "7k.both"
bpe_model_save_path = "res/bpe_models/"
bpe_model_path = f"{bpe_model_save_path}/{bpe_model_name}"
bpe_operations = 10

# dictionaries:
dict_source_path = "res/dictionaries/source_dict.pkl"
dict_target_path = "res/dictionaries/target_dict.pkl"
dict_source = load_dictionary(dict_source_path)
dict_target = load_dictionary(dict_target_path)

# batches:
batches_file_name = "Batches_Test"
batches_readable_name = f'{batches_file_name}_readable'
batches_save_path = "res/batches/"

# feed forward neural network settings
ffnn_model_name = "FFNN_TEST_MODEL2"
ffnn_model_save_path = f"res/models/ffnn/created_models/"
ffnn_model_load_path = f"res/models/ffnn/created_models/{ffnn_model_name}"
ffnn_model_weights_path = f"res/models/ffnn/model_weights/{ffnn_model_name}/"
ffnn_model_plot_path = f"res/models/ffnn/model_figures/{ffnn_model_name}.png"
ffnn_translation_path_greedy = f"res/translations/{ffnn_model_name}_hypothesis_greedy_en"
ffnn_translation_path_beam = f"res/translations/{ffnn_model_name}_hypothesis_beam_en"

# model creation
ffnn_neurons = 200
ffnn_embedding_dimension = 100

# model training
ffnn_epochs = 3
ffnn_batch_size = 200
ffnn_print_info_freq = 50
ffnn_show_info = True
ffnn_eval_frequency = 200

# model search (translation)
ffnn_i_max = 40
ffnn_beam_k = 5

# recurrent neural network settings
rnn_model_name = "RNN_TEST_MODEL2"
rnn_model_save_path = "res/models/rnn/created_models/"
rnn_model_load_path = f"res/models/rnn/created_models/{rnn_model_name}"

rnn_model_weights_path = f"res/models/rnn/model_weights/{rnn_model_name}/"
rnn_model_plot_path = f"res/models/rnn/model_figures/{rnn_model_name}.png"
rnn_model_train_plot_path = f"res/models/rnn/model_figures/{rnn_model_name}_training_history.png"
rnn_translation_path_greedy = f"res/translations/{rnn_model_name}_hypothesis_greedy_en"
rnn_translation_path_beam = f"res/translations/{rnn_model_name}_hypothesis_beam_en"

# model creation
rnn_neurons = 256
rnn_embedding_dimension = 128

# model training
rnn_epochs = 50
rnn_batch_size = 200
rnn_dropout = 0.2

# model search (translation)
rnn_i_max = 43
rnn_beam_k = 5
