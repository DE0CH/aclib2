"""
Created on Apr 14, 2015

@author: Katharina Eggensperger and Andre Biedenkapp
"""

import hashlib
import logging
import time

import numpy
from ConfigSpace.read_and_write import pcs

from epm.preprocessing.pre_feature import PreprocessingFeatures
from epm.reader.feature_parser import FeatureParser
from epm.reader.instance_parser import InstanceParser
from epm.reader.matrix_parser import MatrixParser
from epm.reader.performance_parser import PerformanceParser
from epm.experiment_utils.config_space_utils import  \
    get_imputed_config_from_dict, encode_config_as_array_with_true_values, \
    get_categorical_from_cs, one_hot_encode_array


def load_data(csv_file, feature_file, pcs_file, cutoff, downsample,
              normalize_parameters, instance_file=None, normalize_features=True,
              oneHot=True, par=1, debug=False, _dtype=numpy.float32,
              return_file_id=False, impute_with='default', quality=False):
    """
    reads in csv, feature and pcs file. Returns encoded_matrix, inst_feat_dict,
    inst_list, n_feats_used, perf_list, timeout_list, cen_list, (id_list)
    Does normalize features, impute non-active params, normalize data, and
    one-hot-encoding

    Parameters
    ----------
    csv_file : str
        file with configurations
    feature_file : str
        file with instance features
    pcs_file : str
        file with configurations
    cutoff : int, float
         cutoff for this experiment
    downsample : int,None
        return only that many samples; value is also used as seed
    normalize_parameters : bool
        normalize features (set to false, to allow further data splitting
    instance_file : str
        file with instance subset
    normalize_features : bool
        whether to normalize the features or not
    oneHot : bool
        Perform one-hot encoding for configurations
    par :
        [IGNORED]
    debug : bool
        set logging level to debug
    _dtype : datatype
         all returned values will have this dtype
    return_file_id : bool
        if set to true returns an additional list with file ids for each entry
    impute_with : str
        impute inactive params with 'default' or outlier
    quality : bool
        y is quality so timeout_list, cen_list are NONE

    Return
    ------


    """
    logger = logging.getLogger("epm.experiment_utils.data_handling.load_data")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.info("Quality: %s" % str(quality))

    # Read performance data
    logging.info("Load csv: %s" % csv_file)
    start = time.time()
    pp = PerformanceParser(debug=debug, par=par, cutoff=cutoff, quality=quality)

    if not (isinstance(csv_file, list) or isinstance(csv_file, tuple)):
        csv_file = list([csv_file, ])

    if quality:
        config_list, perf_list, inst_list, id_list = \
            pp.read_data_from_multiple_files_with_id(files_=csv_file)
    else:
        config_list, perf_list, inst_list, timeout_list, cen_list, id_list = \
            pp.read_data_from_multiple_files_with_id(files_=csv_file)

    if downsample is not None and downsample < len(perf_list):
        logger.critical("DOWNSAMPLE from %d to %d" % (len(perf_list),
                                                      downsample))
        rs = numpy.random.RandomState(downsample)
        idx = list(range(len(perf_list)))
        rs.shuffle(idx)
        idx = idx[:downsample]
        config_list = [config_list[i] for i in idx]
        inst_list = [inst_list[i] for i in idx]
        perf_list = [perf_list[i] for i in idx]
        id_list = [id_list[i] for i in idx]
        if not quality:
            timeout_list = [timeout_list[i] for i in idx]
            cen_list = [cen_list[i] for i in idx]

    dur = time.time() - start
    logger.info("Reading performance data took %5.3fsec" % dur)

    # Read features
    start = time.time()
    if feature_file is not None:
        logger.info("Load features: %s" % feature_file)
        fp = FeatureParser()
        inst_feat_dict, n_feats_used = fp.read_features(
                feature_file=feature_file,
                n_feats=-1)

        # Normalize features to be within [0,1]
        if normalize_features:
            start = time.time()
            fpre = PreprocessingFeatures(inst_feats_dict=inst_feat_dict)
            inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)
            dur = time.time() - start
            logger.info("Normalizing features took %5.3fsec" % dur)
        else:
            logger.info("Do not normalize features")
    else:
        logging.info("Fake features")
        inst_feat_dict = dict()
        inst_set = set(inst_list)
        for i in inst_set:
            inst_feat_dict[i] = numpy.array([len(inst_feat_dict), ])
        n_feats_used = 1

    dur = time.time() - start
    logger.info("Getting features took %5.3fsec" % dur)

    # If we want to use a subset of instances given in a file
    if instance_file is not None:
        ip = InstanceParser()
        use_inst_list = ip.read_instances(instance_file)
        # Rebuild all lists
        tmp_config_list = list()
        tmp_perf_list = list()
        tmp_timeout_list = list()
        tmp_cen_list = list()
        tmp_inst_list = list()
        for idx, inst in enumerate(inst_list):
            if inst in use_inst_list:
                tmp_config_list.append(config_list[idx])
                tmp_perf_list.append(perf_list[idx])
                if quality:
                    tmp_timeout_list.append(timeout_list[idx])
                    tmp_cen_list.append(cen_list[idx])
                tmp_inst_list.append(inst_list[idx])
        logger.info("Ignore %d instances due to %s" % (len(perf_list) -
                                                       len(tmp_perf_list),
                                                       instance_file))
        if len(tmp_config_list) == 0:
            raise ValueError("No performances left, maybe wrong instance "
                             "file? %s" % instance_file)
        perf_list = tmp_perf_list
        config_list = tmp_config_list
        inst_list = tmp_inst_list
        if not quality:
            timeout_list = tmp_timeout_list
            cen_list = tmp_cen_list
            assert (len(timeout_list) == len(perf_list) ==
                    len(config_list) == len(cen_list))
        else:
            assert len(perf_list) == len(config_list)

    else:
        logger.info("Use all instances")

    # Build Configuration data
    config_matrix, idx_to_keep = \
        create_config_matrix(config_list=config_list,
                             pcs_file=pcs_file,
                             logger=logger,
                             normalize=normalize_parameters,
                             one_hot=oneHot,
                             _dtype=_dtype,
                             impute_with=impute_with)

    if len(config_matrix) != len(inst_list):
        logger.info('Some configurations have been discarded. '
                    'Therefore, remove also the discarded entries from: '
                    'instance list, performance list, timeout list, '
                    'censored list, id list')
        inst_list = list(numpy.array(inst_list)[idx_to_keep])
        perf_list = list(numpy.array(perf_list)[idx_to_keep])
        timeout_list = list(numpy.array(timeout_list)[idx_to_keep])
        cen_list = list(numpy.array(cen_list)[idx_to_keep])
        id_list = list(numpy.array(id_list)[idx_to_keep])

    logger.info("Finish: Data %s; Feat [%d, %d]" %
                (str(config_matrix.shape), len(inst_feat_dict.keys()),
                 len(list(inst_feat_dict.values())[0])))

    perf_list = numpy.array([_dtype(i) for i in perf_list], dtype=_dtype)

    if quality:
        cen_list = None
        timeout_list = None

    if return_file_id:
        return config_matrix, inst_feat_dict, inst_list, n_feats_used,\
               perf_list, timeout_list, cen_list, id_list
    else:
        return config_matrix, inst_feat_dict, inst_list, n_feats_used, \
               perf_list, timeout_list, cen_list


def loadMatlabData(pcs_file, feature_file, performance_file, config_file,
                   normalize_features=True, max_y=300, oneHot=True,
                   quality=False, normalize_parameters=True):
    """
    reads in csv, feature, performance and config file file. Returns X and y.
    Does normalize features, impute non-active params, normalize data, and
    one-hot-encoding

    Parameters
    ----------
    pcs_file : str
        file with configurations
    feature_file : str
        file to the feature definitions
    performance_file : str

    config_file : str
        matlab file with configurations
    normalize_features : bool
        flag to normalize features
    max_y : float, optional

    oneHot : bool, optional
        enable one hot encoding
    quality : bool, optional
        True if ml scenario
    normalize_parameters : bool, optional
        flag to normalize configurations to interval [0, 1]

    Returns
    -------
    config_matrix, inst_feat_dict, inst_list, n_feats_used, perf_list,
    timeout_list, cen_list, best_perf
    """

    if quality:
        raise ValueError("Reading quality data from matlab is not yet"
                         " implemented")

    logger = logging.getLogger("epm.experiment_utils.data_handling."
                               "loadMatlabData")

    # Read in features
    logging.info("Load features: %s" % feature_file)
    start = time.time()
    fp = FeatureParser()
    inst_feat_dict, n_feats_used = fp.read_features(feature_file=feature_file,
                                                    n_feats=-1)
    dur = time.time() - start
    logger.info("Reading Features took %ssec" % dur)

    # Normalize features to be within [0,1]
    if normalize_features:
        start = time.time()
        fpre = PreprocessingFeatures(inst_feats_dict=inst_feat_dict)
        inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)
        dur = time.time() - start
        logger.info("Normalizing features took %5.3fsec" % dur)
    else:
        logger.info("Do not normalize features")

    # Read in performance data
    logging.info("Load performance_Matrix: %s" % performance_file)
    start = time.time()
    mp = MatrixParser()
    perf_list, timeout_list, cen_list, inst_list, best_perf, ic = \
        mp.read_performance_matrix(performance_file, matrix_captime=max_y)

    dur = time.time() - start
    logger.info("Reading performance data took %ssec" % dur)
    
    logging.info("Load config_matrix: %s" % config_file)
    start = time.time()
    config_list = mp.read_config(config_file, ic[0])
    dur = time.time() - start
    logger.info("Reading config data took %ssec" % dur)

    # Build Configuration data
    config_matrix = create_config_matrix(config_list=config_list,
                                         pcs_file=pcs_file,
                                         logger=logger,
                                         normalize=normalize_parameters,
                                         one_hot=oneHot,
                                         _dtype=numpy.float32,
                                         impute_with='default')

    perf_list = list([numpy.float32(i) for i in perf_list])

    return config_matrix, inst_feat_dict, inst_list, n_feats_used, perf_list, \
        timeout_list, cen_list, best_perf


def partition_data(data_matrix, inst_list, rs=None, debug=True):
    """
    split data into four parts
    :param data_matrix: encoded data matrix with configs
    :param inst_list: list of instances
    :param rs: optional, seed for numpy.random
    :param debug: optional, if True don't retry
    :return: idx for upper_left, upper_right, lower_left, lower_right
    """
    logger = logging.getLogger("epm.experiment_utils.data_handling."
                               "partition_data")

    if rs is None:
        rs = numpy.random.RandomState()

    unique_config = set()
    for row in data_matrix:
        hasher = hashlib.md5()
        hasher.update(row.view(numpy.float32))
        unique_config.add(hasher.hexdigest())
    unique_config = list(unique_config)

    rs.shuffle(unique_config)
    logger.info("Found %d (%d) unique configs" % (len(unique_config),
                                                  data_matrix.shape[0]))

    # get unique instances
    unique_instance = numpy.unique(inst_list)
    rs.shuffle(unique_instance)
    logger.info("Found %d (%d) unique instances" % (len(unique_instance),
                                                    len(inst_list)))

    # Split into two equal parts
    unique_instance = numpy.array_split(unique_instance, 2)
    # unique_config = numpy.array_split(unique_config, 2)
    unique_config = [unique_config[:int(len(unique_config)/2)],
                     unique_config[int(len(unique_config)/2):]]
    # (conf, inst)
    # 00 01
    # 10 11
    indices = list([[[], []], [[], []]])

    # Iterate over matrix and instances
    for idx, row in enumerate(data_matrix):
        # c_hash = row.dot(random_matrix)
        hasher = hashlib.md5()
        hasher.update(row.view(numpy.float16))
        c_hash = hasher.hexdigest()
        if c_hash in unique_config[0]:
            if inst_list[idx] in unique_instance[0]:
                indices[0][0].append(idx)
            elif inst_list[idx] in unique_instance[1]:
                indices[0][1].append(idx)
            else:
                raise ValueError("Could not find instance %s" % inst_list[idx])
        elif c_hash in unique_config[1]:
            if inst_list[idx] in unique_instance[0]:
                indices[1][0].append(idx)
            elif inst_list[idx] in unique_instance[1]:
                indices[1][1].append(idx)
            else:
                raise ValueError("Could not find instance %s" % inst_list[idx])
        else:
            raise ValueError("Could not find config: %s, idx: %d, hash: %s" %
                             (str(row), idx, c_hash))

    # Do some checks
    if len(indices[0][0]) == 0 or len(indices[1][0]) == 0 or \
            len(indices[0][1]) == 0 or len(indices[1][1]) == 0:
        raise ValueError("Could not split, retry with different seed")
    else:
        return indices


def separate_data_with_bools(data_matrix, inst_list, perf_list, succ_list):
    logger = logging.getLogger("epm.experiment_utils.data_handling."
                               "separate_data_with_bools")

    # Remove censored data
    start = time.time()

    succ_list = numpy.array(succ_list)
    true_encoded_matrix = data_matrix[succ_list, :]
    true_perf_list = [perf_list[i] for i, suc in enumerate(succ_list) if suc]
    true_inst_list = [inst_list[i] for i, suc in enumerate(succ_list) if suc]

    false_encoded_matrix = data_matrix[~succ_list, :]
    false_perf_list = [perf_list[i]
                       for i, suc in enumerate(succ_list) if not suc]
    false_inst_list = [inst_list[i]
                       for i, suc in enumerate(succ_list) if not suc]

    dur = time.time() - start
    logger.info("Splitting data took %ssec, true dim: %s, false dim: %s" %
                (dur, str(true_encoded_matrix.shape),
                 str(false_encoded_matrix.shape)))

    return true_encoded_matrix, true_inst_list, true_perf_list, \
        false_encoded_matrix, false_inst_list, false_perf_list


def build_data(data_matrix, inst_list, inst_feat_dict, n_feats):
    # builds numpy array from various data structures
    train_X = numpy.zeros([data_matrix.shape[0],
                           data_matrix.shape[1] + n_feats],
                          dtype=numpy.float32)
    for i in range(data_matrix.shape[0]):
        vec = numpy.hstack((data_matrix[i, :], inst_feat_dict[inst_list[i]]))
        train_X[i] = vec
    return train_X


def warp(x, quality):
    assert type(quality) is bool
    if quality:
        return x
    else:
        return numpy.log10(x)


def unwarp(x, quality):
    assert type(quality) is bool
    if quality:
        return x
    else:
        return numpy.power(10, x)


def create_config_matrix(config_list, pcs_file,
                         logger=None, normalize=False, one_hot=False,
                         _dtype=numpy.float32, impute_with='def'):

    """
    Helper function to create a numerical representation of a list of
    configurations from a given configuration space.
    Impute nonactive params and normalize hyperparameter if set.

    Parameters
    ----------
    config_list
    pcs_file
    logger
    normalize
    one_hot
    _dtype
    impute_with

    Returns
    -------
    np.array
        Number of configuration X number of hyperparameter

    np.array
        Indices of values in config_list which are discarded
    """

    logger = logger or \
        logging.getLogger("epm.experiment_utils.data_handling.load_data")

    start = time.time()

    # Load the configuration space from file
    cs = pcs.read(open(pcs_file, 'r'))
    config_matrix = numpy.zeros((len(config_list),
                                 len(cs.get_hyperparameter_names())),
                                dtype=_dtype)

    indx_to_delete = []
    for indx, config in enumerate(config_list):
        # Create a configuration from a dictionary. A configuration created
        # with the old configuration space module(2015 from aclib), may
        # contain inactive hyperparameters.Therefore, some preprocessing is
        # necessary.
        try:
            config = get_imputed_config_from_dict(cs=cs,
                                                  config=config,
                                                  impute_with=impute_with)

        # array representation scales all continuous parameters between 0-1.
        # Therefore, a rescaling to the original values is necessary.
        # This is done by overwriting the values in the cfg_array with its
        # corresponding non-scaled value.
            cfg_array = encode_config_as_array_with_true_values(config=config,
                                                            cs=cs,
                                                            normalize=normalize)
            config_matrix[indx] = cfg_array

        except Exception as e:
            indx_to_delete.append(indx)

    all_idx = numpy.arange(len(config_list))
    idx_to_keep = numpy.setdiff1d(all_idx, indx_to_delete)
    config_matrix = config_matrix[idx_to_keep]
    logger.info('USE {} of {} configs (Discard {})'
                .format(len(idx_to_keep), len(all_idx),
                        len(all_idx) - len(idx_to_keep)))

    dur = time.time() - start
    logger.info("Converting data to matrix took %5.3fsec, dimensionality: %s" %
                (dur, str(config_matrix.shape)))

    # One hot encode configurations
    start = time.time()
    if one_hot:
        cat_size = get_categorical_from_cs(cs=cs)
        config_matrix = one_hot_encode_array(X=config_matrix,
                                             cat_list=cat_size)
    else:
        config_matrix = config_matrix
    dur = time.time() - start
    logger.info("OneHot-Encoding data took %5.3fsec, d  imensionality: %s" %
                (dur, str(config_matrix.shape)))

    config_matrix = config_matrix.astype(dtype=_dtype)
    return config_matrix, idx_to_keep
