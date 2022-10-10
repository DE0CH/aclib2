import numpy as np
import logging
from time import time
try:
    import cPickle as pickle
except ImportError:
    import pickle
from json_tricks import load as json_load
from pathlib import Path
from epm.experiment_utils.data_handling import warp, unwarp
from epm.models import external_rfr_predict, external_rfr_quantile
from epm.models.external_rfr import load_model
from ConfigSpace.read_and_write import pcs, json as cs_json_load


class SurrogateModel:
    """
    The surrogate model is the interface to the actual model and makes
    predictions.  It receives its queries from the flask-server.

    HINT: Loading the pickled instances and model may take a lot of time and
          memory!
    """
    def __init__(self, **kwargs):
        self.debug = kwargs.get('debug', True)
        
        self.logger = logging.getLogger('EPM Server SurrogateModel')
        if self.debug:
            logging.getLogger().setLevel(level=logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(level=logging.INFO)
            self.logger.setLevel(logging.INFO)
        
        self.logger.debug('Surrogate Model started')
        self.check_kwargs(kwargs)
        self.logger.debug('kwargs checked')

        # TODO: TO WRAPPER, PYRFR, CONFIGSPACE, INST_FEAT_DICT
        #self.pkl_filename = kwargs.get('pkl_filename')
        self.wrapper_filename = Path(kwargs.get('pyrfr_wrapper'))
        self.pyrfr_filename = Path(kwargs.get('pyrfr_model'))
        self.config_space_filename = Path(kwargs.get('config_space'))
        self.inst_feat_dict_filename = Path(kwargs.get('inst_feat_dict'))

        self.quality = kwargs.get('quality')
        self.impute_with = kwargs.get('impute_with')
        self.idle_time = int(float(kwargs.get('idle_time')))
        self.dtype = kwargs.get('dtype')

        self.model = None
        self.inst_feat_dict = None
        self.cs = None

        self.encode = False

        self.STATE = 'OK'
        self.logger.debug('Init passed')

    def load_model(self):

        start_time = time()
        self.logger.debug('Start loading pickled model...')
        try:
            m = load_model(self.wrapper_filename.parent,
                           self.pyrfr_filename,
                           self.wrapper_filename)

            inst_feat_dict = json_load(open(str(self.inst_feat_dict_filename), 'r'))

            if self.config_space_filename.suffix == '.pcs':
                cs = pcs.read(open(str(self.config_space_filename), 'r'))
            elif self.config_space_filename.suffix == '.json':
                cs = cs_json_load.read(open(str(self.config_space_filename), 'r'))
            else:
                cs = pcs.read(open(str(self.config_space_filename), 'r'))
        except Exception as e:
            self.logger.exception(e)
            raise e

        # m, inst_feat_dict, cs = pickle.load(open(self.pkl_filename, 'rb'))

        self.logger.info('Binary Random Forest, Instance Feature Dictionary and'
                         ' ConfigurationSpace successfully loaded.')
        self.logger.debug(f'Loading data takes: {time() - start_time}')

        self.model = m
        self.inst_feat_dict = inst_feat_dict
        self.cs = cs

        if isinstance(self.model, external_rfr_predict.ExternalRFRPredict) or \
                isinstance(self.model,
                           external_rfr_quantile.ExternalRFRQuantile):
            self.logger.critical("NO one-hot encoding")
            self.encode = False
        else:
            self.logger.critical("Activate one-hot encoding")
            self.encode = True

        return self.model, self.inst_feat_dict, self.cs

    def predict(self, X, quality, cutoff=-1.0, quantile_seed=0):
        if isinstance(self.model, external_rfr_predict.ExternalRFRPredict):
            pred = self.model.predict(X, logged=True)
        elif isinstance(self.model, external_rfr_quantile.ExternalRFRQuantile):
            pred = self.model.predict(X, seed=quantile_seed, num_samples=1)
        else:
            pred = self.model.predict(X)
        self.logger.debug("[log10(PRED)]: (%s) -> %s (%s)" % (str(X.shape),
                                                              str(pred),
                                                              str(len(pred))))

        pred = unwarp(pred, quality=quality)
        additional = np.array([])

        if cutoff > 0:
            if len(pred) == 1:
                additional = np.array("TRUE")
                if pred.item() > cutoff:
                    self.logger.critical("[WAR]: Predicted {:.4f}, which is "
                                         "higher than cutoff {}. "
                                         "Set prediction to cutoff"
                                         .format(pred.item(), cutoff))
                    additional = np.array("CUTOFF")
                    # prediction is a single value. Therefore, set it to an
                    # an array containing the cutoff value.
                    pred = np.array([cutoff])
            else:
                # TODO: Write test cases
                additional = np.where(pred >= cutoff, 'SAT', 'CUTOFF')
                # prediction is already an array. Just set each value, which is
                # larger than the cutoff value to the cutoff value
                pred[pred >= cutoff] = cutoff
        self.logger.debug('Prediction: {}\n'
                          'Additional: {}'.format(pred, additional))
        return pred, additional

    def check_kwargs(self, kwargs):
        """
        Check the keyword arguments for completeness

        Args:
            kwargs: dict - keyword arguments from the commandline
        """
        self.logger.debug('Checking kwargs: {}'.format(kwargs))

        if "pyrfr_wrapper" not in kwargs:
            raise ValueError("\'pyrfr_wrapper\' is not in kwargs")
        if not Path(kwargs["pyrfr_wrapper"]).exists():
            self.logger.error("%s does not exist" % kwargs["pyrfr_wrapper"])
            raise FileNotFoundError("%s does not exist"
                                    % kwargs["pyrfr_wrapper"])

        if "pyrfr_model" not in kwargs:
            raise ValueError("\'pyrfr_model\' is not in kwargs")
        if not Path(kwargs["pyrfr_model"]).exists():
            self.logger.error("%s does not exist" % kwargs["pyrfr_model"])
            raise FileNotFoundError("%s does not exist" % kwargs["pyrfr_model"])

        if "config_space" not in kwargs:
            raise ValueError("\'config_space\' is not in kwargs")
        if not Path(kwargs["config_space"]).exists():
            self.logger.error("%s does not exist" % kwargs["config_space"])
            raise FileNotFoundError("%s does not exist"
                                    % kwargs["config_space"])

        if "inst_feat_dict" not in kwargs:
            raise ValueError("\'inst_feat_dict\' is not in kwargs")
        if not Path(kwargs["inst_feat_dict"]).exists():
            self.logger.error("%s does not exist" % kwargs["inst_feat_dict"])
            raise FileNotFoundError("%s does not exist"
                                    % kwargs["inst_feat_dict"])

        if "idle_time" not in kwargs:
            self.logger.error("'idle_time' is not in kwargs")
            raise ValueError("'idle_time' is not in kwargs")

        if "impute_with" not in kwargs:
            self.logger.error("'impute_with' is not in kwargs")
            raise ValueError("'impute_with' is not in kwargs")

        if "dtype" not in kwargs:
            self.logger.error("'dtype' is not in kwargs")
            raise ValueError("'dtype' is not in kwargs")
