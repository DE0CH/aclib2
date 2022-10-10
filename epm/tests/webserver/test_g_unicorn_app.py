from epm.webserver.g_unicorn_app import EpmGUnicornApplication
from epm.webserver.flask_server import create_app
from epm.webserver.flask_server_helper import \
    retrieve_host_port, parse_args, store_credentials
from pathlib import Path


def test_g_unicorn_app():
    print(Path.cwd())

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

    args, _ = parse_args(args)
    print(args.dir)
    kwargs = vars(args)
    kwargs['dont_load_pkl'] = True

    ip, port = retrieve_host_port(args.nic_name,
                                  args.ip,
                                  args.port)
    store_credentials(ip,
                      port,
                      args.pid,
                      args.dir)

    application = create_app(**kwargs)

    options = {'bind': '{}:{}'.format(ip, port),
               'daemon': True,
               'name': kwargs.get('pid')}

    e = EpmGUnicornApplication(application, options)

    # e.run()
    # e.halt()
