"""
Script to start the surrogate webserver from the epm script, as well as
functions to send requests.

Surrogate Communicator Procedure:
    1) Parse parameters
    2) Start webserver + wait a given amount of time
    3) Potentially check whether server is running
    4) Send params to the server
    5) Process results and print the results (--> will be written to file)
"""
import os
import sys
import logging
import subprocess
import numpy as np
from pathlib import Path
from time import time, sleep

from epm.webserver.flask_worker import send_procedure
from epm.webserver.flask_worker_helper import retrieve_credentials, \
    check_if_running, wait_until_running

logging.basicConfig(filename='./epm.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d '
                           '%(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('EPM Surrogate Communicator')


def parse_args(args, debug=True):
    """
    Reads in the command line arguments and parse them.

    Examples
    --------
    >>> args = ['wrapper', 'model', 'config', 'inst_feat', 'i_name', 'i_info', 0.0, 0.0, -1, 40, 'True']
    >>> parsed_args, _ = parse_args(args)
    >>> parsed_args
    {'quality': False, ..., 'instance_name': 'i_name', 'instance_info': 'i_info', 'cutoff': 0.0, 'run_length': 0.0, 'seed': -1, 'idle_time': 40, debug: True, 'dir': ..., 'host': None, 'port': None, 'pid': None}

    >>> args = ['QUALITY', 'wrapper', 'model', 'config', 'inst_feat', 'i_name', 'i_info', 0.0, 0.0, -1, 40, 'False']
    >>> parsed_args, params = parse_args(args)
    >>> parsed_args
    {'quality': True, ..., 'instance_name': 'i_name', 'instance_info': 'i_info', 'cutoff': 0.0, 'run_length': 0.0, 'seed': -1, 'idle_time': 40, debug: False, 'dir': ..., 'host': None, 'port': None, 'pid': None}
    >>> params
    []

    Parameters
    -----------
    args : list(strings)
        Command line arguments

    Returns
    -------
    dict
        A dictionary storing the information
            quality - bool
                whether it is a ML model or not
            pyrfr_wrapper - str
                Path to pyrfr wrapper file
            pyrfr_model - str
                Path to pyrfr model file
            config_space -str
                Path to ConfigSpace file
            inst_feat - str
                Path to instance-feature-dictionary
            instance_name - str
                name of the instance which references the instance in the
                feat dict
            instance_info - str
                additional information to the instance (often not used)
            cutoff - float
                cutoff time
            run_length - float

            seed - int

            idle_time - int
                Time the server is allowed to be idle before it will be killed
            debug - bool
                Flag to activate debug messages
            dir - str
                the current working directory
            host - str

            port - int

            pid - str
                a unique id for the surrogate run

    """
    quality = args[0].upper() == 'QUALITY'
    credential_path = Path.cwd()
    if quality:
        args.pop(0)
    try:
        host, port, pid = retrieve_credentials(credential_path)
    except FileNotFoundError:
        host, port, pid = None, None, None

    parsed_args = {
        'quality': quality,
        'pyrfr_wrapper': args[0],
        'pyrfr_model': args[1],
        'config_space': args[2],
        'inst_feat_dict': args[3],
        'instance_name': args[4],
        'instance_info': args[5],
        'cutoff': float(args[6]),
        'run_length': float(args[7]),
        'seed': int(args[8]),
        'idle_time': int(args[9]),
        'debug': str(args[10]).lower() == 'true',
        'dir': str(credential_path),
        'host': host,
        'port': port,
        'pid': pid
        }
    # Add the additional (algorithm's) parameters into a list.
    try:
        params = args[11:]
    except IndexError:
        params = []

    if debug:
        logger.debug('Parsed Args are: {}'.format(parsed_args))

    return parsed_args, params


def parsed_args_to_list(parsed_args):
    """
    Convert a dictionary to a with parameter names list. By keeping the order.
    This list can be used as command line arguments for the webserver start
    call.


    Examples
    --------
    >>> parsed_args =  {'quality': True, 'pyrfr_wrapper': 'w', 'pyrfr_model': 'm',
    ...                 'config_space': 'c', 'inst_feat_dict': 'i',
    ...                 'instance_name': 'i_name', 'instance_info': 'i_info',
    ...                 'cutoff': 0.0, 'run_length': 0.0, 'seed': -1,
    ...                 'idle_time': 40, 'debug': True,
    ...                 'host': None, 'port': None, 'pid': None}
    >>> list_args = parsed_args_to_list(parsed_args)
    >>> list_args[13]
    '--quality'
    >>> list_args[0]
    '--pyrfr_wrapper'
    >>> list_args[1]
    'w'
    >>> len(list_args)
    14


    Parameters
    ----------
    parsed_args: dict
        Dictionary containing the parameters for the surrogate experiment

    Returns
    -------
    list[str]
        list representation of the dictionary
    """
    command = ['--pyrfr_wrapper',  str(parsed_args.get('pyrfr_wrapper')),
               '--pyrfr_model',    str(parsed_args.get('pyrfr_model')),
               '--config_space',   str(parsed_args.get('config_space')),
               '--inst_feat_dict', str(parsed_args.get('inst_feat_dict')),
               '--dir',            str(parsed_args.get('dir')),
               '--idle_time',      str(parsed_args.get('idle_time')),
               # TODO: PID AND STORE DIRECTORY CURRENTLY FIX
               '--pid',            str(parsed_args.get('pid') or 12345),
               ]

    if parsed_args.get('quality'):
        command.append('--quality')
    if parsed_args.get('debug'):
        command.append('--debug')

    command.append('start')

    logger.debug('Parsed Args in list representation: {}'.format(command))

    return command


def prepare_args_for_sending(args):
    """
    Basically same behavior like parsed_args_to_list. But the args are
    parsed so that they match the command line syntax of the flask worker.
    They can later than be parsed with the flask worker parser and used for
    sending to the surrogate webserver.

    Parameters
    ----------
    args : dict
        parsed arguments

    Returns
    -------
    list[str]
        arguments in list representation with name
    """

    parsed_args_dict, params = parse_args(args)

    parsed_args_list = \
        ['--dir',           str(parsed_args_dict.get('dir')),
         '--instance_name', parsed_args_dict.get('instance_name'),
         '--instance_info', parsed_args_dict.get('instance_info'),
         '--cutoff',        str(parsed_args_dict.get('cutoff')),
         '--run_length',     str(parsed_args_dict.get('run_length')),
         '--seed',          str(parsed_args_dict.get('seed')),
         ]

    logger.debug('Parsed args for sending: {}'
                 .format(parsed_args_list + params))

    return parsed_args_list + params


def acquire_lock(directory):
    """
    Check if a server is running. A server instance is created, if no lock file
    is found and no nameserver credentials are found. This surrogate wrapper
    may be called concurrently from multiple processes. Therefore, it is
    important to make sure, that only one server instance is running.

    Creating a file in the following manner is a atomic operation and therefore
    should suffice as a lock file.
    (See https://stackoverflow.com/questions/33223564/
    atomically-creating-a-file-if-it-doesnt-exist-in-python)

    To support the unambiguity each worker will wait a random amount of time.

    Parameters
    ----------
    directory : Path-like object, str
        path to the lockfile

    Returns
    -------
    bool
        True if the process is able to acquire the lock!
    """
    sleep(np.random.uniform(low=0.000, high=0.0001))
    try:
        lock_file = os.path.join(directory, 'lock.file')
        os.open(lock_file, os.O_CREAT | os.O_EXCL)
        return True
    except FileExistsError:
        return False


def start_server(args):
    """
    Starts the server in a new process.

    Parameters
    ----------
    args : list[str]
        the parsed args in list representation.
    """
    logger.info('python -m epm.webserver.g_unicorn_app + args: {}'.format(args))
    subprocess.Popen(["python", "-m", "epm.webserver.g_unicorn_app"] + args,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)


def check_if_list_return_first(obj):
    """
    Parse the returned values from the surrogate model.
    Normally only single point predictions should be returned. It may occur that
    a list is returned. In this case only the first value will be picked!

    Furthermore, the state of the prediction is extracted.

    Parameters
    ----------
    obj - list[int, float, str], int, float, str
        Either the surrogate model's prediction or a string representing the
        predictions state (e.g. Cutoff,..)

    Returns
    -------
    str, float
        the prediction or the prediction qualifier.
    """
    if obj is None:
        raise ValueError('SurrogateCommunicator: received object is None')

    if type(obj) is list:
        if len(obj) != 1:
            logger.warning('Multiple values in the returned object. '
                           'The first element will be returned\n {}'
                           .format(obj))
        obj = obj[0]

    if type(obj) in [int, float, np.float, np.int]:
        return float(obj)
    elif type(obj) is str:
        obj = obj.upper()
        # TODO: Pm: "status" : <"SAT"/"UNSAT"/"TIMEOUT"/"CRASHED"/"ABORT">
        #       Those are possible outcome! Perhaps I can update the output
        #       signature of the request according to those states!
        if obj == 'CUTOFF':
            logger.critical('Prediction was cut off')
            return 'TIMEOUT'
        elif obj == 'SAT' or obj == 'SATISFIABLE':
            return 'SAT'
        elif obj == 'UNSATISFIABLE':
            return 'UNSAT'
        else:
            return 'SAT'
    else:
        raise ValueError('Type of object {] should be string, int, or float,'
                         'but is {]'.format(obj, type(obj)))


def main(args):
    # First, parse all argument. And prepare the parameters for sending to the
    # webserver.
    parsed_args, params = parse_args(args, debug=False)

    logger.setLevel(
            logging.DEBUG if parsed_args.get('debug') else logging.INFO)
    logging.getLogger().setLevel(
            (logging.DEBUG if parsed_args.get('debug') else logging.INFO))

    logger.debug('Parsed Args are: {}'.format(parsed_args))

    parsed_args_list = parsed_args_to_list(parsed_args)
    parsed_args_sending = prepare_args_for_sending(args)

    # Start the server if no other process is about to start one.
    if acquire_lock(parsed_args.get('dir')) \
            and not check_if_running(directory=parsed_args.get('dir')):
        logger.info('Lock acquired.')
        start_server(args=parsed_args_list)

    wait_until_running(directory=parsed_args.get('dir'),
                       timeout=parsed_args.get('idle_time', 100))

    start_time = time()
    runtime, additional = send_procedure(args=parsed_args_sending)
    dur = time() - start_time
    logging.debug('Sending config and receiving result took {:4f}s'.format(dur))

    # Parse the received results
    runtime = check_if_list_return_first(runtime)
    additional = check_if_list_return_first(additional)
    quality = 1 if parsed_args.get('quality', False) else 0

    # The results will be written to a file. They will be read in by the
    # algorithmWrapper (wrapper.py)
    return_str = "Result for ParamILS:" \
                 "{},{},{},{},{:.2f},{},{}".format(additional,
                                                   runtime,
                                                   quality,
                                                   parsed_args.get('seed'),
                                                   dur,
                                                   parsed_args.get('host'),
                                                   parsed_args.get('port'))
    print('\n', return_str)

    return runtime, additional


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        logger.exception(e)
        logger.exception('Passed args where: {}'.format(sys.argv))
        raise e
