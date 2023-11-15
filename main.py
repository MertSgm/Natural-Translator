"""
This file contains all imports and necessary functions to test this projects' implementations.
To change parameters, simply edit the configuration file located at 'src/config'.
For faster calculations, 'res/data/evaluate' and 'res/data/train' contain two small datasets
for BPE encoding and dictionary creation. For full testing, the original datasets (multi30k.x) can be used.

To execute a function, simply uncomment the import statement and the function call.
All parameters have been pre-adjusted so that they can be tested immediately.

To translate a .txt file with german text, execute the 'beam_search_rnn' function with source_path
being the path of the file.
"""

# import src.config as conf
# from src.metric.bleu import calculate_bleu
# import src.preprocessing.bpe as bpe
# import src.preprocessing.dictionary as dict
# import src.preprocessing.batches as batches
# from src.ffnn.create_ffnn import create_ffnn
# from src.ffnn.train_ffnn import train_ffnn
# from src.ffnn.search_ffnn import greedy_search_ffnn, beam_search_ffnn
# from src.rnn.create_rnn import create_rnn
# from src.rnn.train_rnn import train_rnn
# from src.rnn.search_rnn import greedy_search_rnn, beam_search_rnn

# -- BLEU --
# calculate_bleu(file_path_ref=conf.data_path_evaluate_target,
#                file_path_hyp=conf.ffnn_translation_path)
# calculate_bleu(file_path_ref=conf.data_path_evaluate_target,
#                file_path_hyp=conf.rnn_translation_path)

# -- BPE --
# bpe.create_bpe_model(model_name=conf.example_bpe_model_name,
#                      save_path=conf.bpe_model_save_path,
#                      n=conf.bpe_operations,
#                      file_path_train_data=conf.example_train_source,
#                      file_path_train_data_two=conf.example_train_target)
# bpe.encode_data(file_path_bpe_model=conf.example_bpe_model_path,
#                 file_path_input=conf.example_evaluate_target,
#                 file_path_output=conf.example_bpe_encode_output)
# bpe.decode_data(file_path_input=conf.example_bpe_encode_output,
#                 file_path_output=conf.example_bpe_decode_output)
# bpe.make_bpe_model_readable(file_path_bpe_model=conf.example_bpe_model_path,
#                             file_path_output=conf.example_bpe_readable_output_target)


# The following functions use the original datasets. Calculations might take longer.
# -- Dictionary --
# dict.create_dictionary(name="source_dict", file_path_input=conf.data_path_train_bpe_source)
# dict.create_dictionary(name="target_dict", file_path_input=conf.data_path_train_bpe_target)


# -- Batches --
# entries = batches.create_batches(source_path=conf.data_path_train_bpe_source,
#                                  target_path=conf.data_path_train_bpe_target,
#                                  dict_source=conf.dict_source,
#                                  dict_target=conf.dict_target,
#                                  batch_size=100)
# batches.save_batch_to_file(name=conf.batches_file_name, output_path=conf.batches_save_path, batch_entries=entries)
# batches.save_batch_to_readable_file(name=conf.batches_readable_name,
#                                     output_path=conf.batches_save_path,
#                                     batch_entries=entries,
#                                     source_dict=conf.dict_source,
#                                     target_dict=conf.dict_target)

# -- Feed Forward Neural Network --
# create_ffnn(model_name=conf.ffnn_model_name,
#             save_path=conf.ffnn_model_save_path,
#             plot_save_path=conf.ffnn_model_plot_path,
#             embedding_dim=conf.ffnn_embedding_dimension,
#             neurons=conf.ffnn_neurons,
#             source_dict=conf.dict_source,
#             target_dict=conf.dict_target
#             )
# train_ffnn(model_path=conf.ffnn_model_load_path,
#            save_weights_path=conf.ffnn_model_weights_path,
#            epochs=conf.ffnn_epochs,
#            batch_size=conf.ffnn_batch_size,
#            print_eval_info_freq=conf.ffnn_print_info_freq,
#            show_eval_info=conf.ffnn_show_info,
#            eval_freq=conf.ffnn_eval_frequency,
#            train_source_path=conf.data_path_train_bpe_source,
#            train_target_path=conf.data_path_train_bpe_target,
#            eval_source_path=conf.data_path_evaluate_bpe_source,
#            eval_target_path=conf.data_path_evaluate_bpe_target,
#            source_dict=conf.dict_source,
#            target_dict=conf.dict_target
#            )
# greedy_search_ffnn(model_path=conf.ffnn_model_load_path,
#                    model_weights_path=conf.ffnn_model_weights_path,
#                    source_path=conf.data_path_evaluate_bpe_source,
#                    save_translation_path=conf.ffnn_translation_path_greedy,
#                    source_dict=conf.dict_source,
#                    target_dict=conf.dict_target,
#                    i_max=conf.ffnn_i_max,
#                    )
# beam_search_ffnn(model_path=conf.ffnn_model_load_path,
#                  model_weights_path=conf.ffnn_model_weights_path,
#                  source_path=conf.data_path_evaluate_bpe_source,
#                  save_translation_path=conf.ffnn_translation_path_beam,
#                  source_dict=conf.dict_source,
#                  target_dict=conf.dict_target,
#                  i_max=conf.ffnn_i_max,
#                  k=conf.ffnn_beam_k)


# -- Recurrent Neural Network --
# create_rnn(model_name=conf.rnn_model_name,
#            save_path=conf.rnn_model_save_path,
#            plot_save_path=conf.rnn_model_plot_path,
#            embedding_dim=conf.rnn_embedding_dimension,
#            neurons=conf.rnn_neurons,
#            dropout=conf.rnn_dropout,
#            source_dict=conf.dict_source,
#            target_dict=conf.dict_target)
# train_rnn(model_path=conf.rnn_model_load_path,
#           save_weights_path=conf.rnn_model_weights_path,
#           save_plot_path=conf.rnn_model_train_plot_path,
#           batch_size=conf.rnn_batch_size,
#           epochs=conf.rnn_epochs,
#           train_source_path=conf.data_path_train_bpe_source,
#           train_target_path=conf.data_path_train_bpe_target,
#           eval_source_path=conf.data_path_evaluate_bpe_source,
#           eval_target_path=conf.data_path_evaluate_bpe_target,
#           source_dict=conf.dict_source,
#           target_dict=conf.dict_target)
# greedy_search_rnn(model_path=conf.rnn_model_load_path,
#                   model_weights_path=conf.rnn_model_weights_path,
#                   source_path=conf.data_path_evaluate_bpe_source,
#                   save_translation_path=conf.rnn_translation_path_greedy,
#                   source_dict=conf.dict_source,
#                   target_dict=conf.dict_target,
#                   i_max=conf.rnn_i_max)
# beam_search_rnn(model_path=conf.rnn_model_load_path,
#                 model_weights_path=conf.rnn_model_weights_path,
#                 source_path=conf.data_path_evaluate_bpe_source,
#                 save_translation_path=conf.rnn_translation_path_beam,
#                 source_dict=conf.dict_source,
#                 target_dict=conf.dict_target,
#                 batch_size=conf.rnn_batch_size,
#                 i_max=conf.rnn_i_max,
#                 k=conf.rnn_beam_k)
