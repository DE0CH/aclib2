import sys
import numpy as np
from .flask_worker_helper import parse_args,\
    retrieve_credentials, send_request
import logging

logger = logging.getLogger('EPM Worker')


def send_procedure(args):
    """
    Send a task to solve to the surrogate model.
    A task consist of a
        instance_name - name of the instance
        instance_info - optional
        cutoff - cutoff time
        run_length -
        seed   - random seed number
        params - algorithm parameters

    Parameters
    ----------
    args - parameters with the task's information

    Returns
    -------
    result : float, list[float]
        The surrogate's predictions
    additional : str, list[str]
        a string for each result value, indicating its state (cutoff,...)
    """
    parsed_args, params = parse_args(args)

    host, port, pid = retrieve_credentials(parsed_args.dir)

    json_data = {'instance_name': parsed_args.instance_name,
                 'instance_info': parsed_args.instance_info,
                 'cutoff': parsed_args.cutoff,
                 'run_length': parsed_args.run_length,
                 'seed': parsed_args.seed,
                 'params': params
                 }

    response = send_request(host=host, port=port, json_data=json_data)

    if response == 'FAIL' or response.status_code not in ['200 OK', 200]:
        if type(response) is str:
            logger.error(response)
        else:
            try:
                logger.error(response.json())
            except:
                logger.error(f'No connection established. '
                             f'Response was type {type(response)}.\n'
                             f'Response was {response}')

        # TODO (pm): Should it throw an exception or keep running with some
        #            error return value
        raise AssertionError('Connection not successful')

    # TODO (pm): try-catch? Are there any possibilities, that the aclib can
    #            handle broken runs?
    response = response.json()
    runtime = response.get('result', np.array([-1234]))
    additional = response.get('additional', ['CRASHED'])

    return runtime, additional


if __name__ == "__main__":
    send_procedure(sys.argv[1:])
