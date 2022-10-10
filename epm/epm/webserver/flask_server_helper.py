import argparse
import sys
import pathlib
import pickle
import numpy as np
import logging

from epm.experiment_utils.config_space_utils import \
    encode_config_as_array_with_true_values, one_hot_encode,\
    get_imputed_config_from_dict

logger = logging.getLogger('EPM Server')


def handle_request(json_data, surrogate_model):
    """
    Convert the received json data to a vector representation. Furthermore, a
    instance is looked up by a given instance name from the request.
    Those information are passed to a statistical model, the surrogate model,
    to perform a prediction.

    Parameters
    ----------
    json_data : json
        contains the instance name, which is then looked up in the
        instance_feature_dictionary. It contains also information about the
        cutoff, runlength, seed, and further parameters.

    surrogate_model : epm.surrogates.surrogate_model.SurrogateModel
        This model is used to make a prediction.

    Returns
    -------
    np.array, np.array
        The predictions, as well as a array containing details, whether the
        model has predicted a cutoff or not. In case of no cutoff the value is
        "TRUE", otherwise it is "CUTOFF".
    """

    instance_name = json_data.get('instance_name')
    instance_info = json_data.get('instance_info')
    cutoff = json_data.get('cutoff')
    runlength = json_data.get('runlength')  # ignored
    seed = json_data.get('seed')
    params = json_data.get('params')  # further params passed to the wrapper

    try:
        _ = surrogate_model.inst_feat_dict[instance_name]
    except KeyError:
        logger.warning("Did not found instance in instance-feature-dictionary."
                       " Trying to fix automatically by renaming the instance.")
        # Test if either the instance or the instances in the feature dictionary
        # consist of more parts, e.g. (found for probsat7Sat90) Instance name is
        # 'unif-k7-r85.00-v90-c7650-S227014642.cnf but corresponding entry in
        # feat. dict is:
        # instances/sat/data/7SAT90-SAT-CSSC14/instances/Sat_Data/
        # jack_instances/7sat90.train/unif-k7-r85.00-v90-c7650-S227014642.cnf
        instance_name_ = instance_name.split("/")[-1]
        count = 0
        for candidate in surrogate_model.inst_feat_dict:
            if instance_name_ in candidate:
                count += 1
                instance_name = candidate
                logger.debug('Found instance {} --> Rename to {}'
                             .format(instance_name_, candidate))
                if count >= 2:
                    msg = 'For a test instance more than one ' \
                          'corresponding instance was found in the ' \
                          'instance-feature dictionary. Instance name ' \
                          'should be unique. Please fix those naming ' \
                          'errors and restart the procedure. Instance ' \
                          'name which has occurred multiple ' \
                          'times: {}'.format(instance_name)
                    logger.error(msg)
                    raise ValueError(msg)
    try:
        features = surrogate_model.inst_feat_dict[instance_name]
        features = features.reshape((-1,))
    except KeyError:
        logger.error('Instance not known!\n'
                     'Instance name {}'.format(instance_name))
        surrogate_model.STATE = 'FAIL'
        return np.array([-12345]), np.array("CRASHED")

    params = convert_params_to_vec(params,
                                   surrogate_model.cs,
                                   surrogate_model.encode,
                                   surrogate_model.impute_with
                                   )
    params = params.reshape((-1, ))

    x = np.hstack((params, features)).reshape((1, -1)).astype(surrogate_model.dtype)

    try:
        pred, additional = surrogate_model.predict(
                X=x,
                quality=surrogate_model.quality,
                cutoff=cutoff,
                quantile_seed=seed
                )
    except Exception as e:
        logger.error("While making a prediction the following error occurred:"
                     " {}.\n Arguments are: Params:  {} \n"
                     "Quality: {} Cutoff: {} Seed: {}"
                     .format(e, params, surrogate_model.quality,
                             cutoff, seed))

        return np.array([-12345]), np.array("CRASHED")

    return pred, additional


def convert_params_to_vec(params, cs, encode=True, impute_with='def'):
    """
    Convert str - parameters in list representation to a numerical array
    representation.
    The types of the parameters are determined by the configuration space.
    It is also possible to one-hot-encode categorical features.
    Missing parameters are imputed with the default value.
    # TODO (pm, 15.6): I actually dont know if other imputation types are
                       possible. Please have a look at the ConfigSpace.utils.
    The returned array is in the same form and order as the data the surrogate
    model has been trained on.

    Parameters
    ----------
    params : list(str)
        List of parameters in form '--name value' or '-name value'.
    cs : ConfigSpace.configuration_space
        Contains the information about the parameters from `params`. Parameter
        in `cs` and `params` must match!
    encode : bool
        One-hot encode categorical parameters or not. Defaults to True.
    impute_with : str
        Impute missing values with the default parameter if 'def'.

    Returns
    -------
        np.array
            array representation of the parameters.
    """
    p_dict = {}

    # Test if the named parameter are
    # starting with a '--' instead of a single '-'.
    for i in range(0, len(params), 2):
        if not params[i].startswith('-'):
            msg = 'Parameter not interpretable {}.' \
                  ' It should start with at least one \'-\'' \
                .format(params[i])
            logger.error(msg)
            raise ValueError(msg)

        if params[i].startswith('-') and not params[i].startswith('--'):
            params[i] = '-' + params[i]

        try:
            key = params[i].replace('--', '').strip()
            value = params[i+1]
            p_dict[key] = value
        except IndexError as e:
            logger.error('Index error {}\n'
                         'Could not parse parameters, due to missing value for'
                         'parameter {}'.format(e, params[i])
                         )
            raise e
        except AttributeError as e:
            logger.error('Parameter is type {} but should be str'
                         .format(type(params[i]))
                         )
            raise e

    # Create a configuration from the configuration space according to cs
    # But first make sure, the data types match the data types from the
    # configuration space
    config = get_imputed_config_from_dict(config=p_dict,
                                          cs=cs,
                                          impute_with=impute_with)

    logger.debug('Config before encoding: {}'.format(config))
    if encode:
        config = one_hot_encode(config, cs)
        logger.debug('Config as array in one-hot-encoding: {}'.format(config))
    else:
        # TODO (pm): internal array representation or non-normalized values?
        # this one is scaled between 0 and 1.
        config = encode_config_as_array_with_true_values(config=config,
                                                         cs=cs)
        logger.debug('Config as array without one-hot-encoding {}'
                     .format(config))
    return config


def retrieve_host_port(nic_name=None, ip=None, port=0):
    """
    Creates from a network interface name (e.g. 'localhost') a valid ip
    address and a port. If no ip and no nic name is specified, the default
    network address will be picked.

    Parameters
    ----------
    nic_name : str, None
    ip : int, None
    port : int

    Returns
    -------
        ip, port
    """
    import socket

    if ip is None:
        if nic_name is None:
            try:
                host = socket.gethostbyname(socket.getfqdn())
            except socket.gaierror:
                host = socket.gethostbyname(socket.gethostname())
        elif nic_name.lower() == 'localhost':
            host = 'localhost'
        else:
            raise ValueError('nic_name is not specified')
    else:
        host = ip

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    ip, port = sock.getsockname()
    sock.close()
    logger.debug('Retrieved host {} on port {}'.format(ip, port))
    return ip, port


def store_credentials(host, port, pid, working_dir):
    """
    Stores the server credentials (ip and port) in a file.

    Parameters
    ----------
    host : int
    port : int
    pid : int
        Process id
    working_dir : str, Pathlike-object
    """
    working_dir = pathlib.Path(working_dir)
    with open(str(working_dir / 'nameserver_creds.pkl'), 'wb') as f:
        pickle.dump((host, port, pid), f)
    logger.debug('Stored server credentials in directory {}'
                 .format(str(working_dir / 'nameserver_creds.pkl')))


def parse_args(args):
    """
    Parses the commandline arguments for the daemon to start.

    Returns
    -------
        parsed args
    """
    parser = argparse.ArgumentParser(description='Start/Stop daemon',
                                     prog='EPM')
    parser.add_argument('--pyrfr_wrapper', dest='pyrfr_wrapper', default=None,
                        required=True, help='Path to Pyrfr Wrapper')
    parser.add_argument('--pyrfr_model', dest='pyrfr_model', default=None,
                        required=True,
                        help='Path to binary Random Forest (pyrfr)')
    parser.add_argument('--config_space', dest='config_space', default=None,
                        required=True, help='Path to config space file')
    parser.add_argument('--inst_feat_dict', dest='inst_feat_dict', default=None,
                        required=True, help='Path to instances-feature-'
                                            'dictionary in json format')
    parser.add_argument('--dir', dest='dir', default='./tmp', required=False,
                        help='Store daemon related files here')
    parser.add_argument('--pid', dest='pid', default=None, required=True,
                        help='Unique number', type=int)
    parser.add_argument('--idle_time', dest='idle_time', default=60 * 10,
                        required=False, type=int,
                        help='Idle time before surrogate daemon dies')
    parser.add_argument('--impute_with', dest='impute_with', default='def',
                        help='How to impute nonactive parameter')
    parser.add_argument('--quality', dest='quality', default=False,
                        action='store_true', help='Handle quality data?')
    parser.add_argument('--dtype', dest='dtype', default=np.float32,
                        required=False,
                        help='Optional: casts input data to the surrogate model'
                             ' to a certain format. (Default: numpy.float32)')
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true', help='Show some more output')

    subparser = parser.add_subparsers(dest='action')
    start = subparser.add_parser('start')
    start.add_argument('-n', '--nic_name', default=None, type=str,
                       dest='nic_name',
                       help='Nic name of the server. E.g. localhost')
    start.add_argument('--ip', default=None, type=str, dest='ip',
                       help='Ip address for daemon')
    start.add_argument('-p', '--port', default=0, type=int, dest='port',
                       help='Port of the daemon to use')

    subparser.add_parser('stop')
    subparser.add_parser('status')

    args, unknown = parser.parse_known_args(args)

    if args.action == 'start' \
            and args.nic_name is not None and args.ip is not None:
        raise AttributeError('Nicname and ip address is given!'
                             ' Only ip address will be taken into account')

    return args, unknown


def parse_args_state_stop(args):
    """
    Similar to argument parse but only used for checking the state and send shutdown signals.

    :param args:
    :return:
    """
    parser = argparse.ArgumentParser(description='Check/Stop daemon',
                                     prog='EPM')
    parser.add_argument('--dir', dest='dir', default=str(pathlib.Path.cwd()), required=False,
                        help='Directory of the server information.')
    subparser = parser.add_subparsers(dest='action')
    subparser.add_parser('stop')
    subparser.add_parser('status')
    args, unknown = parser.parse_known_args(args)
    return args, unknown


if __name__ == '__main__':
    args, unknown = parse_args(sys.argv[1:])
