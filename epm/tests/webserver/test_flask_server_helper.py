import pytest
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import pathlib
import tempfile
from epm.webserver.flask_server_helper import retrieve_host_port, \
    parse_args, store_credentials, convert_params_to_vec
from epm.experiment_utils.config_space_utils import one_hot_encode
from epm.webserver.flask_worker_helper import retrieve_credentials


def test_retrieve_host_port():
    nic_name_test = 'localhost'
    port_test = 5050

    ip, port = retrieve_host_port(nic_name=nic_name_test,
                                  ip=None, port=port_test)
    assert ip == '127.0.0.1'
    assert port == port_test

    # random port + no error
    _, _ = retrieve_host_port(None, None, 0)

    # fixed ip and no localhost
    ip, port = retrieve_host_port(None, '127.0.0.1', port_test)
    assert ip == '127.0.0.1'
    assert port == port_test


def test_store_credentials():
    with tempfile.TemporaryDirectory() as tmp_dir:
        nic_name_test = 'localhost'
        port_test = 5050
        pid = 1234

        ip, port = retrieve_host_port(nic_name_test, None, port_test)

        store_credentials(ip, port, pid, tmp_dir)
        assert (pathlib.Path(tmp_dir) / 'nameserver_creds.pkl').exists()

        restored_ip, restored_port, restored_pid = \
            retrieve_credentials(pathlib.Path(tmp_dir))
        assert ip == restored_ip
        assert port == restored_port
        assert pid == restored_pid

        restored_ip, restored_port, restored_pid = \
            retrieve_credentials(tmp_dir)
        assert ip == restored_ip
        assert port == restored_port
        assert pid == restored_pid


def test_parse_args():
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
            '--idle', str(60),
            'start',
            '--port', str(0)
            ]
    parsed_args, unknown = parse_args(args)

    assert parsed_args.action == 'start'
    assert parsed_args.nic_name is None
    assert parsed_args.ip is None
    assert parsed_args.port == 0
    assert parsed_args.dtype == np.float32

    args = ['--pyrfr_wrapper',
            model_path + 'pyrfr_wrapper.cplex_rcw.par10.random.pkl',
            '--pyrfr_model',
            model_path + 'pyrfr_model.cplex_rcw.par10.random.bin',
            '--config_space',
            model_path + 'config_space.cplex_rcw.par10.random.pcs',
            '--inst_feat_dict',
            model_path + 'inst_feat_dict.cplex_rcw.par10.random.json',
            '--pid', '12345',
            '--idle', str(60),
            'start',
            'stop',
            '--port', str(0)
            ]
    parsed_args, unknown = parse_args(args)

    assert parsed_args.action == 'start'
    assert parsed_args.action != 'stop'
    assert parsed_args.nic_name is None
    assert parsed_args.ip is None
    assert parsed_args.port == 0

    args = ['--pyrfr_wrapper',
            model_path + 'pyrfr_wrapper.cplex_rcw.par10.random.pkl',
            '--pyrfr_model',
            model_path + 'pyrfr_model.cplex_rcw.par10.random.bin',
            '--config_space',
            model_path + 'config_space.cplex_rcw.par10.random.pcs',
            '--inst_feat_dict',
            model_path + 'inst_feat_dict.cplex_rcw.par10.random.json',
            '--pid', '12345',
            '--idle', str(60),
            'stop'
            ]
    parsed_args, unknown = parse_args(args)

    assert parsed_args.action == 'stop'
    with pytest.raises(AttributeError):
        assert parsed_args.nic_name is None
    with pytest.raises(AttributeError):
        assert parsed_args.ip is None
    with pytest.raises(AttributeError):
        assert parsed_args.port is None


def test_parse_args_server_address():
    model_path = '../files/test_model/'
    args = ['--pyrfr_wrapper',
            model_path + 'pyrfr_wrapper.cplex_rcw.par10.random.pkl',
            '--pyrfr_model',
            model_path + 'pyrfr_model.cplex_rcw.par10.random.bin',
            '--config_space',
            model_path + 'config_space.cplex_rcw.par10.random.pcs',
            '--inst_feat_dict',
            model_path + 'inst_feat_dict.cplex_rcw.par10.random.json',
            '--pid', '15',
            '--action', 'start',
            '--nic_name', 'localhost',
            '--port', 0]

    parsed_args, unknown = parse_args(args)
    assert parsed_args.action == 'start'
    assert parsed_args.nic_name == 'localhost'
    assert parsed_args.ip is None
    assert parsed_args.port == 0

    args = ['--pyrfr_wrapper',
            model_path + 'pyrfr_wrapper.cplex_rcw.par10.random.pkl',
            '--pyrfr_model',
            model_path + 'pyrfr_model.cplex_rcw.par10.random.bin',
            '--config_space',
            model_path + 'config_space.cplex_rcw.par10.random.pcs',
            '--inst_feat_dict',
            model_path + 'inst_feat_dict.cplex_rcw.par10.random.json',
            '--pid', '15',
            '--action', 'start',
            '--nic_name', 'localhost',
            '--ip', '127.0.0.1'
            '--port', 0]

    with pytest.raises(AttributeError):
        _, _ = parse_args(args)


def test_one_hot_encoding():

    cs = CS.ConfigurationSpace(seed=1)
    a = CSH.UniformIntegerHyperparameter('a', lower=10, upper=100, log=False)
    b = CSH.CategoricalHyperparameter('b', choices=['red', 'green', 'blue'])

    cs.add_hyperparameters([a, b])
    cfg = cs.sample_configuration()

    c = one_hot_encode(cfg, cs)

    assert np.array_equal(np.array([[47, 0, 0, 1]]), c)
    assert c.shape == (1, 4)


def test_convert_params():
    cs = CS.ConfigurationSpace(seed=1)
    a = CSH.UniformIntegerHyperparameter('a', lower=10, upper=100, log=False)
    b = CSH.CategoricalHyperparameter('b', choices=['red', 'green', 'blue'])
    cs.add_hyperparameters([a, b])

    c = convert_params_to_vec(['--a', '15', '--b', 'red'], cs, encode=True)
    assert np.array_equal(c, np.array([[15.,  0.,  0.,  1.]]))

    c = convert_params_to_vec(['--a', '15', '--b', 'blue'], cs, encode=False)
    assert np.array_equal(c, np.array([15, 2]))

    c = convert_params_to_vec(['--a', '15', '--b', 'blue'], cs, encode=False,
                              impute_with='def')
    assert np.array_equal(c, np.array([15, 2]))
