import pathlib
import argparse
import requests
import logging
from time import sleep
from json import JSONDecodeError
from functools import wraps
try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger('EPM Worker')


def check_server_status_decorator(func):
    """
    A simple decorater function. Same functionality as check_server_status.
    Checks before executing a function, if a server is running.
    The function should have the parameters host and port either as
    named arguments or on first or second argument position.

    Parameters
    ----------
    func : func
        Function to be decorated

    Returns
    -------
    func
        The decorated functions output.
    """

    @wraps(func)
    def wrapper_check_server_status(*args, **kwargs):
        if kwargs.get('host') is not None and kwargs.get('port') is not None:
            host, port = kwargs.get('host'), kwargs.get('port')
        else:
            if len(args) < 2:
                raise AttributeError(
                        'Could not parse host and port information from'
                        ' function call with parameters: '
                        '\nargs: {},\nkwargs: {}'.format(args, kwargs))

            logger.warning('Try to find host and port arguments from parameter'
                           ' list, not from kwargs.\n Host and Port should be'
                           ' on first and second position in {}'.format(args))
            host, port = args[:2]
        running = check_server_status(host, port)

        return func(*args, **kwargs) if running else "FAIL"

    return wrapper_check_server_status


def check_if_running(directory, throw_exception=False):
    """
    Retrieve the host and port information and check if the server is already
    running.

    Parameters
    ----------
    directory : Path-like object, str
        path to server credentials
    throw_exception : bool, optional
        if set to True, an exception is thrown if the server is not running.

    Returns
    -------
    bool
        state of the server
    """
    try:
        host, port, pid = retrieve_credentials(directory)
    except FileNotFoundError:
        logger.warning('No server credentials found in directory {}.'
                       .format(directory))
        return False

    running = check_server_status(host, port)
    if throw_exception and not running:
        raise TimeoutError('Server is still not running... ')

    return running


def wait_until_running(directory, timeout=200):
    """ Wait to make sure that the server is running """
    time_slept = 0
    while not check_if_running(directory,
                               throw_exception=(time_slept > timeout)):
        logger.debug('SLEEP {:3d} - Running: False'.format(time_slept))
        sleep(10)
        time_slept += 10

    logger.debug('SLEEP {:3d} - Running: True'.format(time_slept + 5))


def check_server_status(host, port, update_last_request=True):
    """
    Sends a request to the server and asks for its current status.
    The server is (hopefully) running on ip `host` and port `port`

    Parameters
    ----------
    host : str
    port : int
    update_last_request : bool
        Flag. If set to True, the last request counter in app will be updated.

    Returns
    -------
    bool
        State of the server.
        True if the server is running, else False
    """
    try:
        request_type = 'status' if update_last_request else 'last_request'
        response = requests.get('http://{}:{}/{}'
                                .format(host, port, request_type),
                                headers={'content_type': 'application/json'})
        status = response.json().get('STATUS') == 'RUNNING'

    except (requests.exceptions.ConnectionError, JSONDecodeError) as e:
        logging.debug('Check server status --> Error: {}'
                      'Server is not running.'.format(type(e)))
        status = False

    logging_level = logger.debug if status else logger.error
    logging_level('Server is' + ('' if status else ' not') +
                  ' running on {}:{}'.format(host, port))
    return status


def retrieve_last_request(host, port):
    """
    Check the last_request time of a server
    (hopefully) running on ip `host` and port `port`

    Parameters
    ----------
    host : int, str
    port : int

    Returns
    -------
    bool
        State of the server.
        True if the server is running, else False
    """
    try:
        response = requests.get('http://{}:{}/last_request'
                                .format(host, port),
                                headers={'content_type': 'application/json'})
        status = response.json().get('STATUS') == 'RUNNING'
        last_request = response.json().get('LAST_REQUEST')
    except (requests.exceptions.ConnectionError, JSONDecodeError) as e:
        logging.debug('retrieve last request --> Error: {}'
                      'Server is not running.'.format(type(e)))
        status = False
        last_request = None

    logging_level = logger.info if status else logger.error
    logging_level('Server is' + ('' if status else ' not') +
                  ' running on {}:{} + Last request was: '
                  '{}'.format(host, port,
                              last_request.__format__('.2f')
                              if last_request is not None else 'Failure'))
    return status, last_request


@check_server_status_decorator
def send_request(host, port, json_data):
    """ Helper-function: Send data to server"""
    logger.debug('Send request to {}:{}'.format(host, port))
    logger.debug('Send data: {}'.format(json_data))

    return requests.post('http://{}:{}/predict'.format(host, port),
                         json=json_data,
                         headers={'content_type': 'application/json'})


@check_server_status_decorator
def send_shutdown_signal(host, port):
    """ Helper-function: Send shutdown signal to server"""
    logger.info('Send shutdown signal to {}:{}'.format(host, port))

    return requests.get('http://{}:{}/shutdown'.format(host, port),
                        headers={'content_type': 'application/json'})


def retrieve_credentials(working_dir):
    """
    Read in the webservers credentials from file.

    Parameters
    ----------
    working_dir : str, Path-like obj
        Path to the directory with the name server credentials.

    Returns
    -------
    str, int, int
        host, port, process id

    """
    try:
        p = working_dir / 'nameserver_creds.pkl'
    except TypeError:
        from os.path import join
        p = join(working_dir, 'nameserver_creds.pkl')

    try:
        host, port, pid = pickle.load(open(str(p), 'rb'))
    except FileNotFoundError as e:
        logger.error(e)
        raise FileNotFoundError(e)

    return host, port, pid


def parse_args(args):
    """
    Function to parse the server args.

    Parameters
    ----------
    args : list[str]

    Returns
    -------
    parsed args, list[str]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dir', default='.',
                        help='Path to nameserver credentials')
    parser.add_argument('--instance_name', dest='instance_name',
                        required=True)
    parser.add_argument('--instance_info', dest='instance_info', required=False)
    parser.add_argument('--cutoff', dest='cutoff', required=True,
                        type=float)
    parser.add_argument('--run_length', dest='run_length', required=False,
                        type=float, default=None)
    parser.add_argument('--seed', dest='seed', required=False,
                        default=0, type=int)
    # Unknown contains all parameters for the random forest etc
    try:
        args, unknown = parser.parse_known_args(args)
    except SystemExit:
        logger.warning(f'There was an error while parsing the arguments. Trying to fix problem automatically. '
                       f'Increase verbosity for more details.')
        logger.debug(f'The error has occurred most likely in the argument-parsing of the send procedure. '
                     f'Perhaps there is an argument in the algorithm\'s parameters, which has '
                     f'the same name as one of the required arguments. Or perhaps an algorithm\'s argument could '
                     f'have the same name as some standard arguments, such as \'-h\'. \n'
                     f'Args are: {args}. \n Trying to fix problem automatically by splitting the parameters in known '
                     f'and unknown and parsing it again.')

        args, unknown = args[:12], args[12:]
        try:
            args = parser.parse_args(args)
        except SystemExit as e:
            logger.exception('Parsing arguments still does not work. SystemExit')
            raise e

    args.dir = pathlib.Path(args.dir)
    return args, unknown


def parse_params(params):
    """
    Parse the parameters for the surrogate model.
    Those parameters may start with `Quality`. This indicates that the scenario
    is a ML scenario.

    Parameters
    ----------
    params : list(str)

    Returns
    -------
    list(str)
    bool
        quality - True if ML scenario else False
    int
        seed
    """
    if params[0] == 'QUALITY':
        quality = True
        params.pop()
    else:
        quality = False

    # Keep seed
    seed = int(float(params[4]))

    return params, quality, seed
