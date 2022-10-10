from pathlib import Path


model_path = Path('../files/test_model/')

test_file_exists = ((model_path / 'pyrfr_wrapper.cplex_rcw.par10.random.pkl').exists()
                    and (model_path / 'pyrfr_model.cplex_rcw.par10.random.bin').exists()
                    and (model_path / 'config_space.cplex_rcw.par10.random.pcs').exists()
                    and (model_path / 'inst_feat_dict.cplex_rcw.par10.random.json').exists())

skip_surrogate_test_reason = 'Tests need a surrogate model in pickled form. ' \
                             'Due to its large size, it is, however,' \
                             ' not included in the repository.'
