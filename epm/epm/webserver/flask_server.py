import sys
import numpy as np
import logging
import pathlib
from time import time
from subprocess import Popen
from flask import Flask, jsonify, request, Response, json
from .flask_server_helper import retrieve_host_port, \
    store_credentials, handle_request, parse_args, parse_args_state_stop
from .flask_worker_helper import retrieve_credentials, check_server_status, \
    send_shutdown_signal
from epm.surrogates.surrogate_model import SurrogateModel

logger = logging.getLogger('EPM Server')


def create_app(**kwargs):
    """ Factory function to create a flask application """
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        MODEL_PATH=kwargs.get('dir'),
        )
    pathlib.Path(app.instance_path).mkdir(exist_ok=True, parents=True)

    logger.info('Start Creating Webserver')
    app.surrogate_model = SurrogateModel(**kwargs)

    # Skip the pkl loading step for some testing scenarios.
    if not kwargs.get('dont_load_pkl', False):
        app.surrogate_model.load_model()
    else:
        logger.debug('Skip loading the Pkl model.'
                     ' This is only activated to speed up testing.')

    app.LAST_REQUEST = time()
    app.IDLE_TIME = kwargs.get('idle_time', 600)
    app.kwargs = kwargs

    cmd = 'python -m epm.webserver.monitor_server --dir {} --idle_time {}'\
        .format(kwargs.get('dir'), app.IDLE_TIME)
    logger.info('Start process for server monitor with cmd: {}'.format(cmd))
    app.watcher_proc = Popen(cmd.split())

    @app.route("/predict", methods=['POST'])
    def predict():
        """
        POST-method to predict the runtime of an algorithm instance.

        Predict given a instance name, instance info, seed, and a cutoff time
        the runtime of a algorithm. This algorithm is represented by the
        surrogate model.

        It is also possible to make a direct prediction given the np.array (1xD)
        representation of the parameter from above.

        .. note::

            The instance name must be included in the pickled
            instance-feature-dictionary, which was used during training.

        """
        app.LAST_REQUEST = time()

        if request.method == 'POST':

            # TODO: catch receive errors. catch jsonify errors.
            try:
                json_data = request.get_json()

                # Sometime json data is not a dict but a string.
                # If it is a string, cast it to a dictionary
                if json_data is str:
                    json_data = json.loads(json_data)

            except Exception as e:
                logger.error(e)
                raise e

            # This is a direct call to make a prediction
            # (without instance feature dict, etc.)
            if 'x' in json_data:
                x = np.array(json_data['x'], dtype=app.surrogate_model.dtype)
                result, additional = \
                    app.surrogate_model.predict(x, app.surrogate_model.quality)

            # The parameters are a list of arguments. Convert them to
            # a vector representation and pass them to the surrogate model.
            else:
                if 'params' not in json_data:
                    raise KeyError('No command line parameters are given')
                result, additional = \
                    handle_request(json_data, app.surrogate_model)

            # Return a response in json format
            response = {'result': result.tolist(),
                        'additional': additional.tolist()}
            return Response(response=json.dumps(response),
                            status=200,
                            mimetype='application/json')

            # except KeyError:
            #     return jsonify('KeyError: Wrong input')
            # except ValueError:
            #     return jsonify('ValueError: Wrong input')
        else:
            message = 'Only POST request is implemented.'
            return jsonify(message)

    @app.route('/status', methods=['GET'])
    def get_status():
        """
        Check the current status of the server.

        Returns
        -------
        Response Object
            with STATUS: 'Running' and the time of the LAST_REQUEST
        """

        app.LAST_REQUEST = time()

        if request.method == 'GET':
            response = {'STATUS': 'RUNNING', 'LAST_REQUEST': app.LAST_REQUEST}
            return Response(response=json.dumps(response),
                            status=200,
                            mimetype='application/json')

    @app.route('/last_request', methods=['GET'])
    def get_last_request():
        """
        Similar to get status, but doesn't update the last request counter!
        """
        if request.method == 'GET':
            response = {'STATUS': 'RUNNING', 'LAST_REQUEST': app.LAST_REQUEST}
            return Response(response=json.dumps(response),
                            status=200,
                            mimetype='application/json')

    @app.route('/shutdown', methods=['GET'])
    def shutdown():
        """
        Shut down the webserver.

        Returns
        -------
        str
        """

        logger.debug('Start Shutdown procedure')

        # kill watcher process
        logger.info('Kill Server Monitor')
        app.watcher_proc.terminate()

        # clean up server files and release lock
        cwd = pathlib.Path(app.kwargs.get('dir', '.'))
        nameserver_file = cwd / 'nameserver_creds.pkl'
        lock_file = cwd / 'lock.file'
        pid_file = cwd / 'gunicorn-{}.pid'.format(app.kwargs.get('pid'))

        # Read in the process id of the gUnicorn server
        pid = pid_file.read_text().rstrip()

        # Delete the server files.
        for file in [pid_file, nameserver_file, lock_file]:
            try:
                file.unlink()
                logger.debug('{} deleted'.format(file.name))
            except FileNotFoundError:
                logger.info('Deletion of {} file not successful.'
                            .format(file.name))

        # Shut down server
        kill_cmd = "sleep 2; kill -15 {}".format(pid)
        logger.info('SERVER SHUTTING DOWN with command {}'.format(kill_cmd))
        Popen(kill_cmd, shell=True)
        logger.info('Server shutdown successful.')

        return jsonify('SERVER SHUTTING DOWN with command {}'.format(kill_cmd))

    return app


if __name__ == "__main__":
    if len(sys.argv) == 4 or len(sys.argv) == 2:
        args, unknown = parse_args_state_stop(sys.argv[1:])
    else:
        args, unknown = parse_args(sys.argv[1:])

    if args.action == 'start':
        ip, port = retrieve_host_port(args.nic_name, args.ip, args.port)
        store_credentials(ip, port, args.pid, args.dir)
        app = create_app(**vars(args))
        app.run(debug=args.debug, host=ip, port=port, threaded=False)

    elif args.action == 'status' or args.action == 'stop':
        host, port, pid = retrieve_credentials(args.dir)
        running = check_server_status(host=host,
                                      port=port,
                                      update_last_request=False)

        print('Server is {} running'.format(('not' if not running else '')))
        if args.action == 'stop':
            send_shutdown_signal(host=host, port=port)

    else:
        print('Action state not known')
