import pytest
from epm.surrogates.surrogate_model import SurrogateModel
from epm.models import external_rfr_quantile
from epm.webserver.flask_server_helper import parse_args
from global_consts import test_file_exists, skip_surrogate_test_reason
from pathlib import Path


@pytest.fixture
def surrogate_model():
    model_path = '../files/test_model/'
    args = ['--pyrfr_wrapper',
            model_path + 'pyrfr_wrapper.cplex_rcw.par10.random.pkl',
            '--pyrfr_model',
            model_path + 'pyrfr_model.cplex_rcw.par10.random.bin',
            '--config_space',
            model_path + 'config_space.cplex_rcw.par10.random.pcs',
            '--inst_feat_dict',
            model_path + 'inst_feat_dict.cplex_rcw.par10.random.json',
            '--pid', '12345',
            '--idle', str(30),
            '--dir', str(Path.cwd().absolute()),
            'start',
            ]
    parsed_args, _ = parse_args(args)
    return SurrogateModel(**vars(parsed_args))


def test_surrogate_model(surrogate_model):

    assert surrogate_model.model is None
    assert not surrogate_model.quality
    assert surrogate_model.impute_with == 'def'
    assert surrogate_model.idle_time == 30


@pytest.mark.skipif(not test_file_exists,
                    reason=skip_surrogate_test_reason)
def test_load_surrogate_model(surrogate_model):
    assert surrogate_model.model is None

    model, inst_feat_dict, cs = surrogate_model.load_model()

    assert isinstance(surrogate_model.model,
                      external_rfr_quantile.ExternalRFRQuantile)
    assert surrogate_model.model is model
    assert surrogate_model.inst_feat_dict is inst_feat_dict
    assert surrogate_model.cs is cs

    # import pprint
    # pprint.pprint(inst_feat_dict.keys())
