import pytest
import numpy as np
from flask import json
from ConfigSpace.read_and_write import pcs
from epm.webserver.flask_server import create_app
from epm.webserver.flask_server_helper import parse_args, handle_request
from global_consts import test_file_exists, skip_surrogate_test_reason


@pytest.fixture
def app(dont_load_pkl=False):
    def get_app(port=0, idle=60, dont_load_pkl=dont_load_pkl):
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
                '--idle', str(idle),
                'start',
                '--port', str(port)
                ]

        parsed_args, _ = parse_args(args)

        kwargs = vars(parsed_args)
        kwargs['dont_load_pkl'] = dont_load_pkl

        a = create_app(**kwargs)
        a.testing = True
        return a
    return get_app


@pytest.fixture
def client(app):
    return app().test_client()


@pytest.fixture
def client_without_pkl(app):
    return app(dont_load_pkl=True).test_client()


@pytest.fixture
def runner(app):
    return app().test_cli_runner()


def test_status(client_without_pkl):
    from time import time
    request_time = time()
    response = client_without_pkl.get('/status')
    assert response.status == '200 OK'

    result = response.get_json()
    assert result.get('STATUS') == 'RUNNING'
    assert type(result.get('LAST_REQUEST')) is float
    assert np.isclose(result.get('LAST_REQUEST'), request_time)


@pytest.mark.skipif(not test_file_exists, reason=skip_surrogate_test_reason)
def test_timer(app):
    app = app()
    client = app.test_client()
    last_request = app.LAST_REQUEST
    for i in range(5):
        _ = client.get('/status')
        current_request_time = app.LAST_REQUEST
        assert current_request_time > last_request
        last_request = current_request_time

"""
def test_shutdown(app):
 
    port = 65523
    flask_app = app(port=port, idle=10)
    client = flask_app.test_client()

    from threading import Thread
    from werkzeug.serving import run_simple

    t = Thread(target=flask_app.run,
               kwargs=dict(debug=False,
                           host='127.0.0.1',
                           port=port,
                           threaded=False
                           ))
    t = Thread(target=run_simple,
               kwargs=dict(hostname='127.0.0.1', port=port,
                           application=flask_app, use_reloader=False))

    t.start()
    print(t.is_alive())
    assert t.is_alive()
    last_request = flask_app.LAST_REQUEST
    assert client.get('/status').get_json().get('STATUS') == 'RUNNING'
    assert flask_app.LAST_REQUEST > last_request
    with pytest.raises(SystemExit):
        client.post('/shutdown')
    sleep(4)
    print(t.is_alive())
    assert not t.is_alive()
"""


@pytest.mark.skipif(not test_file_exists, reason=skip_surrogate_test_reason)
def test_large(client):

    print("\n", 40*"#", "START LARGE TEST", 40*"#", "\n")

    assert client.get('/status').status == "200 OK"

    assert not client.application.surrogate_model.quality \
        and client.application.surrogate_model.impute_with == 'def' \
        and client.application.surrogate_model.idle_time == 60 \
        and client.application.surrogate_model.dtype == np.float32 \
        and client.application.surrogate_model.model is not None \
        and len(client.application.surrogate_model.inst_feat_dict) is not 0 \
        and not client.application.surrogate_model.encode \
        and client.application.surrogate_model.STATE == 'OK'

    # Server is now online
    # Start sending queries
    instance_name = 'instances/mip/data/RCW2/RCW-INSTANCES-2/' \
                    'map1027-s103214-b04-h40-n5.lp'

    params = '--probtype 1.0 --n_vars 82346.0 --n_constr 328818.0 ' \
             '--n_nzcnt 440919.0 --nq_vars 0.0 --nq_constr 0.0 --nq_nzcnt 0.0 '\
             '--lp_avg 0.0379278 --lp_l2_avg 0.0816024 --lp_linf 0.418641 ' \
             '--lp_objval 140.118 --num_b_variables 146.0 --num_i_variables ' \
             '0.0 --num_c_variables 82200.0 --num_s_variables 0.0 ' \
             '--num_n_variables 0.0 --ratio_b_variables 0.00177301 ' \
             '--ratio_i_variables 0.0 --ratio_c_variables 0.998227 ' \
             '--ratio_s_variables 0.0 --ratio_n_variables 0.0 ' \
             '--num_ip_variables 146.0 --ratio_ip_variables 0.00177301 ' \
             '--num_unbounded_disc 0.0 --ratio_unbounded_disc 0.0 ' \
             '--support_size_avg 2.0 --support_size_median 2.0 ' \
             '--support_size_varcoef 0.0 --support_size_q90mq10 0.0 ' \
             '--rhs_c_0_avg 417.678 --rhs_c_0_varcoef 286.019 --rhs_c_1_avg ' \
             '0.0024428 --rhs_c_1_varcoef 20.2081 --rhs_c_2_avg 0.0 ' \
             '--rhs_c_2_varcoef 0.0 --vcg_constr_deg0_avg 0.25043 ' \
             '--vcg_constr_deg0_median 0.0 --vcg_constr_deg0_varcoef 1.94777 ' \
             '--vcg_constr_deg0_q90mq10 1.0 --vcg_var_deg0_avg 564.014 ' \
             '--vcg_var_deg0_median 201.0 --vcg_var_deg0_varcoef 1.43471 ' \
             '--vcg_var_deg0_q90mq10 1200.0 --vcg_constr_weight0_avg 2604.38 ' \
             '--vcg_constr_weight0_varcoef 573.371 --vcg_var_weight0_avg ' \
             '5865520.0 --vcg_var_weight0_varcoef 0.526162 ' \
             '--A_ij_normalized0_avg 0.287671 --A_ij_normalized0_varcoef ' \
             '0.927066 --a_normalized_varcoefs0_avg 3.44072e-05 ' \
             '--a_normalized_varcoefs0_varcoef 573.426 --obj_coefs0_avg 0.0 ' \
             '--obj_coefs0_std 0.0 --obj_coef_per_constr0_avg 0.0 ' \
             '--obj_coef_per_constr0_std 0.0 --obj_coef_per_sqr_constr0_avg ' \
             '0.0 --obj_coef_per_sqr_constr0_std 0.0 --vcg_constr_deg1_avg ' \
             '1.09049 --vcg_constr_deg1_median 1.0 --vcg_constr_deg1_varcoef ' \
             '0.321141 --vcg_constr_deg1_q90mq10 0.0 --vcg_var_deg1_avg ' \
             '4.3622 --vcg_var_deg1_median 4.0 --vcg_var_deg1_varcoef ' \
             '0.170759 --vcg_var_deg1_q90mq10 1.0 --vcg_constr_weight1_avg ' \
             '0.444012 --vcg_constr_weight1_varcoef 1.95368 ' \
             '--vcg_var_weight1_avg 1.77614 --vcg_var_weight1_varcoef ' \
             '0.341319 --A_ij_normalized1_avg 1.0 --A_ij_normalized1_varcoef ' \
             '0.0 --a_normalized_varcoefs1_avg 0.0543797 ' \
             '--a_normalized_varcoefs1_varcoef 4.39085 --obj_coefs1_avg ' \
             '0.0101582 --obj_coefs1_std 0.100274 --obj_coef_per_constr1_avg ' \
             '0.00253954 --obj_coef_per_constr1_std 0.0250686 ' \
             '--obj_coef_per_sqr_constr1_avg 0.00507908 ' \
             '--obj_coef_per_sqr_constr1_std 0.0501372 --vcg_constr_deg2_avg ' \
             '1.34092 --vcg_constr_deg2_median 1.0 --vcg_constr_deg2_varcoef ' \
             '0.418129 --vcg_constr_deg2_q90mq10 1.0 --vcg_var_deg2_avg ' \
             '5.35447 --vcg_var_deg2_median 4.0 --vcg_var_deg2_varcoef ' \
             '7.73612 --vcg_var_deg2_q90mq10 1.0 --vcg_constr_weight2_avg ' \
             '2604.82 --vcg_constr_weight2_varcoef 573.273 ' \
             '--vcg_var_weight2_avg 10401.4 --vcg_var_weight2_varcoef 26.8126 '\
             '--A_ij_normalized2_avg 0.998739 --A_ij_normalized2_varcoef ' \
             '0.0320185 --a_normalized_varcoefs2_avg 0.0544171 ' \
             '--a_normalized_varcoefs2_varcoef 4.40274 --obj_coefs2_avg ' \
             '0.0101401 --obj_coefs2_std 0.100186 --obj_coef_per_constr2_avg ' \
             '0.00253504 --obj_coef_per_constr2_std 0.0250466 ' \
             '--obj_coef_per_sqr_constr2_avg 0.00507007 ' \
             '--obj_coef_per_sqr_constr2_std 0.0500932 --mipgap 1.0 --nodecnt '\
             '0.0 --clqcnt 0.0 --covcnt 0.0 --itcnt_max -512.0 ' \
             '--numnewsolution_sum 0.0 --newin_sum 0.0 --nodeleft_avg -512.0 ' \
             '--nodeleft_varcoef -512.0 --diffObj_avg -512.0 --diffObj_median '\
             '-512.0 --diffObj_varcoef -512.0 --diffObj_q90mq10 -512.0 ' \
             '--numfeas 0.0 --iinf_avg -512.0 --iinf_median -512.0 ' \
             '--iinf_varcoef -512.0 --iinf_q90mq10 -512.0 --diffBestInt_avg ' \
             '-512.0 --diffBestInt_median -512.0 --diffBestInt_varcoef -512.0 '\
             '--diffBestInt_q90mq10 -512.0 --diffBestObjUp_avg -512.0 ' \
             '--diffBestObjUp_median -512.0 --diffBestObjUp_varcoef -512.0 ' \
             '--diffBestObjUp_q90mq10 -512.0 --numcuts_sum 0.0 --diffGap_avg ' \
             '-512.0 --diffGap_median -512.0 --diffGap_varcoef -512.0 ' \
             '--diffGap_q90mq10 -512.0 --pre_t 1.41 --rel_t 3.64 --new_row ' \
             '14361.0 --new_col 6907.0 --new_nonzero 32099.0 --clique_table ' \
             '-512.0 --cliqueCuts 0.0 --impliedBoundCuts 0.0 --flowCuts 0.0 ' \
             '--mixedIntegerRoundingCuts 0.0 --gomoryFractionalCuts 0.0 ' \
             '--time_relax 14.0833 --time_VCG0 0.616667 --time_VCG1 1.35 ' \
             '--time_VCG2 1.38333 --cplex_prob_time 9.26667'
    params = params.split(' ')

    json_data = json.dumps({'instance_name': instance_name,
                            'instance_info': 0,
                            'cutoff': 10000.0,
                            'run_length': 2147483647,
                            'seed': -1,
                            'params': params
                            })

    # 1. Test direct passing to handle request
    out_1, add_1 = \
        handle_request(json.loads(json_data), client.application.surrogate_model)

    # 2. Send it via post request
    response = \
        client.post('/predict', data=json_data, content_type='application/json')
    response = response.get_json()
    out_2, add_2 = response.get('result'), response.get('additional')

    assert np.isclose(out_1, 176.0649, rtol=0.001)
    assert np.isclose(out_2, 176.0649, rtol=0.001)
    assert np.isclose(out_1, out_2, rtol=0.001)
    assert add_1 == add_2

    print("\n", 40*"#", "END LARGE TEST", 40*"#", "\n")


def test_shutdown(client_without_pkl):
    # create gunicorn pid file
    from pathlib import Path
    pid_file = Path('./tmp')
    pid_file.mkdir(parents=True, exist_ok=True)
    pid_file = pid_file / 'gunicorn-12345.pid'
    pid_file.touch(exist_ok=True)
    pid = '123456789'
    pid_file.write_text(pid)

    response = client_without_pkl.get('/shutdown')
    assert response.get_json() == 'SERVER SHUTTING DOWN with ' \
                                  'command sleep 2; kill -15 {}'.format(pid)

    pid_file.parent.rmdir()
