import pytest
import tempfile

from pathlib import Path
from epm.webserver.flask_worker_helper import parse_args, check_if_running, \
    wait_until_running, check_server_status, retrieve_last_request
from epm.webserver.flask_server_helper import store_credentials, \
    retrieve_host_port

from time import time


def test_parse_arg():

    args = ['--dir', '.',
            '--instance_name', 'instances/mip/data/RCW2/'
                               'RCW-INSTANCES-2/map1027-s200538-b04-h40-n5.lp',
            '--instance_info', 'QUALITY',
            '--cutoff', '1000',
            '--seed', '1234',
            '--param1', 'value_param_1', '--param2', 'value_param_2'
            ]

    parsed_args, unknown = parse_args(args)
    assert issubclass(type(parsed_args.dir), Path)
    assert parsed_args.dir.absolute() == Path.cwd()

    assert parsed_args.instance_name == \
        'instances/mip/data/RCW2/RCW-INSTANCES-2/' \
        'map1027-s200538-b04-h40-n5.lp'
    assert parsed_args.instance_info == 'QUALITY'
    assert parsed_args.cutoff == 1000.0
    assert parsed_args.seed == 1234

    assert len(unknown) == 4
    assert unknown[0] == '--param1'
    assert unknown[1] == 'value_param_1'
    assert unknown[2] == '--param2'
    assert unknown[3] == 'value_param_2'


def test_parse_args_2():
    call = "--dir ./out " \
           "--instance_name instances/mip/data/RCW2/RCW-INSTANCES-2/map1027-s200538-b04-h40-n5.lp " \
           "--instance_info 0 " \
           "--cutoff 10000.0 " \
           "--run_length 2147483647 " \
           "--seed -1 " \
           "--barrier_algorithm 0 --barrier_crossover 0 " \
           "--barrier_limits_corrections -1 --barrier_limits_growth 1.0E12 " \
           "--barrier_ordering 0 --barrier_startalg 1 --emphasis_memory no " \
           "--emphasis_mip 0 --emphasis_numerical no --feasopt_mode 0 " \
           "--lpmethod 0 --mip_cuts_cliques 0 --mip_cuts_covers 0 " \
           "--mip_cuts_disjunctive 0 --mip_cuts_flowcovers 0 " \
           "--mip_cuts_gomory 0 " \
           "--mip_cuts_gubcovers 0 --mip_cuts_implied 0 --mip_cuts_mcfcut 0 " \
           "--mip_cuts_mircut 0 --mip_cuts_pathcut 0 " \
           "--mip_cuts_zerohalfcut 0 " \
           "--mip_limits_aggforcut 3 --mip_limits_cutpasses 0 " \
           "--mip_limits_cutsfactor 4.0 --mip_limits_gomorycand 200 " \
           "--mip_limits_gomorypass 0 --mip_limits_submipnodelim 500 " \
           "--mip_ordertype 0 --mip_strategy_backtrack 0.9999 " \
           "--mip_strategy_bbinterval 7 --mip_strategy_branch 0 " \
           "--mip_strategy_dive 0 --mip_strategy_file 1 " \
           "--mip_strategy_fpheur 0 " \
           "--mip_strategy_heuristicfreq 0 --mip_strategy_lbheur no " \
           "--mip_strategy_nodeselect 1 --mip_strategy_presolvenode 0 " \
           "--mip_strategy_probe 0 --mip_strategy_rinsheur 0 " \
           "--mip_strategy_search 0 --mip_strategy_startalgorithm 0 " \
           "--mip_strategy_subalgorithm 0 --mip_strategy_variableselect 0 " \
           "--network_netfind 2 --network_pricing 0 " \
           "--preprocessing_aggregator -1 --preprocessing_boundstrength -1 " \
           "--preprocessing_coeffreduce 2 --preprocessing_dependency -1 " \
           "--preprocessing_dual 0 --preprocessing_fill 10 " \
           "--preprocessing_linear 1 --preprocessing_numpass -1 " \
           "--preprocessing_reduce 3 --preprocessing_relax -1 " \
           "--preprocessing_repeatpresolve -1 --preprocessing_symmetry -1 " \
           "--read_scale 0 --sifting_algorithm 0 --simplex_crash 1 " \
           "--simplex_dgradient 0 --simplex_limits_perturbation 0 " \
           "--simplex_limits_singularity 10 --Cimplex_perturbation_switch no " \
           "--simplex_pgradient 0 --simplex_pricing 0 --simplex_refactor 0 " \
           "--simplex_tolerances_markowitz 0.01"

    args = call.split(' ')
    parsed_args, unknown = parse_args(args)

    assert parsed_args.instance_name == \
        'instances/mip/data/RCW2/RCW-INSTANCES-2/' \
        'map1027-s200538-b04-h40-n5.lp'
    assert parsed_args.instance_info == '0'
    assert parsed_args.cutoff == 10000.0
    assert parsed_args.seed == -1
    assert '' not in unknown and ' ' not in unknown


def test_parse_args_missing():
    args = ['--instance_name', 'instances/mip/data/RCW2/'
                               'RCW-INSTANCES-2/map1027-s200538-b04-h40-n5.lp',
            '--instance_info', 'QUALITY',
            '--cutoff', '1000',
            '--seed', '1234'
            ]
    parsed_args, unknown = parse_args(args)
    assert parsed_args.dir is not None
    assert parsed_args.dir.absolute() == Path.cwd()

    args = ['--instance_info', 'QUALITY',
            '--cutoff', '1000',
            '--seed', '1234'
            ]
    with pytest.raises(TypeError):
        parsed_args(args)

    args = ['--instance_name', 'instances/mip/data/RCW2/'
                               'RCW-INSTANCES-2/map1027-s200538-b04-h40-n5.lp',
            '--instance_info', 'QUALITY',
            '--seed', '1234'
            ]
    with pytest.raises(TypeError):
        parsed_args(args)


def test_multiple_instance_info():
    args = ['--instance_name', 'instances/mip/data/RCW2/'
                               'RCW-INSTANCES-2/map1027-s200538-b04-h40-n5.lp',
            '--instance_info', 'QUALITY',
            '--cutoff', '1000',
            '--seed', '1234'
            ]

    parsed_args, _ = parse_args(args)
    assert parsed_args.instance_info == 'QUALITY'


def test_check_if_running():

    dir = 'non_existing_dir'
    assert not check_if_running(dir)

    with tempfile.TemporaryDirectory() as tmp_dir:
        nic_name_test = 'localhost'
        port_test = 5050
        pid = 1234

        ip, port = retrieve_host_port(nic_name_test, None, port_test)
        store_credentials(ip, port, pid, tmp_dir)
        assert not check_if_running(tmp_dir)

        with pytest.raises(TimeoutError):
            check_if_running(tmp_dir, throw_exception=True)


def test_wait_until_running():

    with tempfile.TemporaryDirectory() as tmp_dir:
        nic_name_test = 'localhost'
        port_test = 5050
        pid = 1234

        ip, port = retrieve_host_port(nic_name_test, None, port_test)
        store_credentials(ip, port, pid, tmp_dir)

        start = time()
        with pytest.raises(TimeoutError):
            wait_until_running(tmp_dir, 5)

        # procedure should wait for 10 seconds
        # (see code: wait for 10 then raise exception)
        assert 9 < time() - start < 11


def test_check_server_status():
    # dummy test without running server
    random_host = '127.0.0.1'
    random_port = 12345

    assert not check_server_status(host=random_host, port=random_port,
                                   update_last_request=True)
    assert not check_server_status(host=random_host, port=random_port,
                                   update_last_request=False)


def test_last_request():
    # dummy test without running server
    random_host = '127.0.0.1'
    random_port = 12345
    state, last_request = retrieve_last_request(host=random_host,
                                                port=random_port)
    assert not state
    assert last_request is None
