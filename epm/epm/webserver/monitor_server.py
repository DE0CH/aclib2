import sys
import argparse
import logging

from pathlib import Path
from time import time, sleep
from epm.webserver.flask_worker_helper import retrieve_credentials, \
    retrieve_last_request, send_shutdown_signal, wait_until_running

logging.basicConfig(filename='./daemon.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s '
                           '%(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger('EPM Server-Monitor')


def start_monitor(args):
    """
    Monitor the servers activity. Send periodically requests to check if the
    server is too long inactive.

    If it is idle for more than its fixed threshold,
    send a shutdown signal. If this process is not able to find the server,
    it will shutdown itself automatically.

    Parameters
    ----------
    args : argparser object
        args containing the directory of the server credentials, and the
        server's idle time threshold
    """
    try:
        # Give the server some time to start up
        wait_until_running(directory=args.dir, timeout=args.idle_time)
        host, port, pid = retrieve_credentials(working_dir=args.dir)
        logger.debug('Connected monitoring successfully '
                     'to server. (Server on {}:{})'
                     .format(host, port))
    except (TimeoutError, FileNotFoundError) as e:
        logger.error(e)
        sys.exit()

    while True:
        status, last_request = retrieve_last_request(host=host, port=port)
        logger.debug('State of the server {} - '
                     'Last Request {:.2f}'
                     .format(status, last_request))

        # If the server is not available, terminate this process
        if status is False or last_request is None:
            logger.error('Server is not reachable')

            # TODO: Delete lockfile? If no server reachable the server has to be
            #       crashed. Else this server monitor could not be still alive.
            lock_file = Path(args.dir) / 'lock.file'
            try:
                lock_file.unlink()
                logger.debug('{} deleted'.format(lock_file.name))
            except FileNotFoundError:
                logger.info('Deletion of {} file not successful.'
                            .format(lock_file.name))
            sys.exit()

        if time() - last_request >= args.idle_time:
            logger.info('Server is inactive for too long. Send shutdown-signal')
            send_shutdown_signal(host=host, port=port)
            sys.exit(1)

        sleep(min(args.idle_time, 30))


def parse_args():
    parser = argparse.ArgumentParser(description='Server Monitor',
                                     prog='EPM')
    parser.add_argument('--dir', type=str, required=True,
                        help='Path to the directory containing the server'
                             ' credentials')
    parser.add_argument('--idle_time', type=int, default=100,
                        help='Idle time in seconds')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_monitor(args)
