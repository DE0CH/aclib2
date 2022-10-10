import re

import numpy
from epm.experiment_utils.config_space_utils import \
    encode_config_as_array_with_true_values, get_imputed_config_from_dict


def read_validationObjectiveMatrix_file(fn):
    """ COPIED FROM pySMAC, modified
    reads the run data of a validation run performed by SMAC.

    For cases with instances, not necessarily every instance is used during the
    configuration phase to estimate a configuration's performance. If validation
    is enabled, SMAC reruns parameter settings (usually just the final
    incumbent) on the whole instance set/a designated test set. The data from
    those runs is stored in separate files. This function reads one of these
    files.

    :param fn: the name of the validationObjectiveMatrix file
    :type fn: str

    :returns: dict -- configuration ids as keys, list of performances on each
    instance as values.

    .. todo::
       testing of validation runs where more than the final incumbent is
       validated
    """
    values = {}

    with open(fn, 'r') as fh:
        header = fh.readline().split(",")
        num_configs = len(header) - 2
        re_string = '\w?,\w?'.join(['"(.*)"', '"(-?\d*)"'] +
                                   ['"([0-9.]*)"'] * num_configs)
        for line in fh.readlines():
            match = (re.match(re_string, line))
            if match.group(1) in values:
                print("Cannot handle more than one seed per instance")
            values[match.group(1)] = \
                list(map(float,
                         list(map(match.group, list(range(3, 3+num_configs))))))
    return values


def read_validationCallStrings_file(fn):
    """ COPIED FROM pySMAC
    Reads a validationCallString file into a list of dictionaries.

    :returns: list of dicts -- each dictionary contains 'parameter name' and
    'parameter value as string' key-value pairs
    """
    param_dict_list = []
    with open(fn, 'r') as fh:
        for line in fh.readlines()[1:]:
            # skip header line
            config_string = line.split(",")[1].strip('"')
            config_string = config_string.split(' ')
            tmp_dict = {}
            for i in range(0, len(config_string), 2):
                tmp_dict[config_string[i].lstrip('-')] = \
                    config_string[i+1].strip("'")
            param_dict_list.append(tmp_dict)
    return param_dict_list


def read_trajectory_file(fn):
    """ COPIED FROM pySMAC, modified to work on validate over time file
    Reads a trajectory file and returns a list of dicts with all the
    information.

    All values, like "Estimated Training Performance" and so on
    are floats.

    :param fn: name of file to read
    :type fn: str

    :returns: list of dicts -- every dict contains the keys:
        "CPU Time Used", "Estimated Training Performance",
        "Wallclock Time", "Incumbent ID","Automatic Configurator (CPU) Time", ..
    """
    return_list = []

    with open(fn, 'r') as fh:
        header = list(map(lambda s: s.strip('"'), fh.readline().split(",")))
        l_info = len(header)-1
        for line in fh.readlines():
            tmp = line.split(",")
            tmp_dict = {}
            for i in range(l_info):
                tmp_dict[header[i]] = float(tmp[i].strip().replace('"', ''))
            return_list.append(tmp_dict)
    return return_list


def read_default_data(val_obj_fl, val_str_fl, pcs, normalize_parameters,
                      impute_with='def'):
    # Read in Default Data
    perf_def = read_validationObjectiveMatrix_file(val_obj_fl)
    def_config = read_validationCallStrings_file(val_str_fl)
    assert len(def_config) == 1
    def_config = def_config[0]

    # Create a configuration from the
    cs = pcs.read(open(pcs, 'r'))
    config = get_imputed_config_from_dict(cs=cs,
                                          config=def_config,
                                          impute_with=impute_with)
    config = \
        encode_config_as_array_with_true_values(config=config,
                                                cs=cs,
                                                normalize=normalize_parameters)

    inst_list = list()
    perf_list = list()
    for inst in perf_def.keys():
        assert len(perf_def[inst]) == 1
        inst_list.append(inst)
        perf_list.append(numpy.float32(perf_def[inst][0]))

    return config, inst_list, perf_list
