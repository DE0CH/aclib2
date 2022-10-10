import sys
import os
from gunicorn import util, debug
from gunicorn.app.base import BaseApplication
from gunicorn.six import iteritems
from .flask_server import create_app
from .flask_server_helper import parse_args, store_credentials,\
    retrieve_host_port
import logging

logging.basicConfig(filename='./daemon.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s '
                           '%(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('EPM Daemon')


class EpmGUnicornApplication(BaseApplication):
    """
    Class to deploy a custom flask wsgi server as gUnicorn server.
    --> http://docs.gunicorn.org/en/stable/custom.html
    """
    def __init__(self, app, options=None):
        """__init__ method

        Load the base config and assign some core attributes.
        """
        self.usage = None
        self.callable = None
        self.options = options or {}
        self.application = app
        super(EpmGUnicornApplication, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

    def run(self):
        # enables a debug trace
        if self.cfg.spew:
            debug.spew()

        # Runs server in background (daemon)
        if self.cfg.daemon:
            # util.daemonize(self.cfg.enable_stdio_inheritance)
            util.daemonize(True)

        # set python paths
        if self.cfg.pythonpath:
            paths = self.cfg.pythonpath.split(",")
            for path in paths:
                pythonpath = os.path.abspath(path)
                if pythonpath not in sys.path:
                    sys.path.insert(0, pythonpath)

        super(EpmGUnicornApplication, self).run()


def run_application(args, daemon=False):
    """
    Deploy a gUnicorn application. The pid is used as process
    name. With the process name, the gUnicorn server is later killed if it is
    inactive for too long.

    Parameters
    ----------
    args : argparser object
    daemon : bool
        Daemonize the server (run it in background) or not.

    Returns
    -------
        None
    """
    # Read in the ip and the port from the command line arguments and store them
    # in a file so that the worker can find the master node.
    logger.info(f'args where: {args}')
    args, unknown = parse_args(args)

    logger.info(f'DEBUG IN DAEMON is {args.debug}.')
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    logger.debug('About to start server. Parsed arguments are:'
                 'known: {} \n and unknown: {}'
                 .format(args, unknown))

    ip, port = retrieve_host_port(args.nic_name, args.ip, args.port)
    store_credentials(ip, port, args.pid, args.dir)

    # For debugging. Omit the pickle loading to speed up.
    # args.dont_load_pkl = True

    # Create the webserver and deploy it with gUnicorn.
    application = create_app(**vars(args))
    options = {'bind':      '{}:{}'.format(ip, port),
               'daemon':    daemon,
               'proc_name': str(args.pid),
               'pidfile':   os.path.join(args.dir, 'gunicorn-{}.pid'.format(args.pid))
               }
    logger.debug("PIDFILE: {}".format(options.get('pidfile')))
    e = EpmGUnicornApplication(application, options)

    logger.info('Start gUnicorn server on {}:{}, Daemon: {},'
                ' process name: {}'
                .format(ip, port, str(daemon), args.pid))
    e.run()
    logger.debug('gUnicorn server started on {}:{}, Daemon: {},'
                 ' process name: {}'
                 .format(ip, port, str(daemon), args.pid))


if __name__ == "__main__":
    try:
        logger.info('Start Gunicorn Server')
        run_application(sys.argv[1:])
    except Exception as e:
        logger.exception(e)
        raise Exception(e)
