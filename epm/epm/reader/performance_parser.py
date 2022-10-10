"""
Created on Mar 25, 2015

@author: manju
"""

import logging
import numpy
import json


class PerformanceParser(object):
    """
    reads collected performance data of target algorithm
    """

    def __init__(self, cutoff, debug=False, par=1, dtype=numpy.float32,
                 quality=False):
        """
        Constructor
        """
        self.logger = logging.getLogger("PerformanceParser")

        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.par = dtype(par)
        self.cutoff = dtype(cutoff)
        self.dtype = dtype
        self.quality = quality
        self.logger.debug("Read quality: %s" % str(self.quality))
        self.logger.debug("Using PAR-%d results (%f*%d)" %
                          (self.par, self.cutoff, self.par))

    def read_data_from_multiple_files(self, files_):
        if not isinstance(files_, list) and not isinstance(files_, tuple):
            raise ValueError("'files_' is not a list")
        config_list = []
        perf_list = []
        instance_list = []
        if self.quality:
            # here we do not have timeouts and censored data
            for fl in files_:
                self.logger.debug("Read %s" % fl)
                cl, pl, il = self.read_quality_data(fl)
                config_list.extend(cl)
                perf_list.extend(pl)
                instance_list.extend(il)
            return config_list, perf_list, instance_list
        else:
            timeout_list = []
            cencored_list = []
            for fl in files_:
                self.logger.debug("Read %s" % fl)
                cl, pl, il, sl, cenl = self.read_data(fl)
                config_list.extend(cl)
                perf_list.extend(pl)
                instance_list.extend(il)
                timeout_list.extend(sl)
                cencored_list.extend(cenl)
            return config_list, perf_list, instance_list, timeout_list, \
                   cencored_list

    def read_data_from_multiple_files_with_id(self, files_):
        if not isinstance(files_, list) and not isinstance(files_, tuple):
            raise ValueError("'files_' is not a list")
        config_list = []
        perf_list = []
        instance_list = []
        file_id = []

        if self.quality:
            for idx, fl in enumerate(files_):
                self.logger.debug("Read %s" % fl)
                cl, pl, il = self.read_quality_data(fl)
                config_list.extend(cl)
                perf_list.extend(pl)
                instance_list.extend(il)
                file_id.extend([idx] * len(cl))
            return config_list, perf_list, instance_list, file_id
        else:
            timeout_list = []
            cencored_list = []
            for idx, fl in enumerate(files_):
                self.logger.debug("Read %s" % fl)
                cl, pl, il, sl, cenl = self.read_data(fl)
                config_list.extend(cl)
                perf_list.extend(pl)
                instance_list.extend(il)
                timeout_list.extend(sl)
                cencored_list.extend(cenl)
                file_id.extend([idx] * len(cenl))
            return config_list, perf_list, instance_list, timeout_list, \
                   cencored_list, file_id

    def read_data(self, file_):
        """
            reads running time data in generic wrapper format (either in old csv
            format or in new json format)
            Checks the ending of the file to decide how to parse the file
            Args:
                file_ : path to either .json or .csv file
        """
        if file_.endswith("csv"):
            return self._read_csv(file_)
        elif file_.endswith("json"):
            return self._read_json(file_)

    def read_quality_data(self, file_):
        """
            reads quality data in generic wrapper format (either in old csv
            format or in new json format)
            Checks the ending of the file to decide how to parse the file
            Args:
                file_ : path to either .json or .csv file
        """
        if file_.endswith("csv"):
            raise NotImplementedError("No implemented reading quality data so "
                                      "far")
            #return self._read_quality_csv(file_)
        elif file_.endswith("json"):
            return self._read_quality_json(file_)

    def _read_csv(self, fn):
        """
            parses data in generic wrapper format "target_algo_runs.csv"
            instance,seed,status,performance,config,[misc]
            Args:
                fn : path to "target_algo_runs.csv"
        """
        timeout_list = []   # timeout runs
        censored_list = []  # censored runs
        config_list = []    # configurations (dictionaries: name->value)
        perf_list = []      # performance
        instance_list = []
        
        with open(fn) as fp:
            fp.readline()
            for line in fp:
                line = line.strip("\n")
                try:
                    parts = line.split(",")
                    instance = parts[0].strip(" ")
                    # seed = parts[1].strip(" ") # not necessary?
                    status = parts[2].strip(" ")
                    performance = numpy.float32(parts[3])
                    config = parts[4:]
                    misc = None

                except IndexError: 
                    # target_algo_runs.csv can have corrupted lines because all
                    # smac runs wrote in the same file
                    self.logger.warning("Could not read line in performance "
                                     "data file: %s" % line)
                    continue
                except ValueError:
                    # target_algo_runs.csv can have corrupted lines because all
                    # smac runs wrote in the same file
                    self.logger.warning("Could not read line in performance "
                                     "data file: %s" % line)
                    continue

                config_dict = {}
                for c in config:
                    try:
                        param, value = c.strip(" ").split("=")
                        # remove leading "-" in param
                        config_dict[param[1:]] = value
                    except ValueError:
                        # if not splitable, not anymore a param-value pair;
                        # probably, misc info
                        misc = c
                        break

                perf, cen, timeout = self.handle_run(instance,
                                                     status,
                                                     performance,
                                                     misc)

                if perf > 0:
                    config_list.append(config_dict)
                    perf_list.append(perf)
                    instance_list.append(instance)
                    timeout_list.append(timeout)
                    censored_list.append(cen)
                
        self.logger.debug("%s: %d configuration runs." % (fn, len(config_list)))
                    
        return config_list, perf_list, instance_list, timeout_list, \
               censored_list
    
    def _read_json(self, fn):
        """
            parses data in generic wrapper format "target_algo_runs.json"
           {"status":...,
           "misc":...,
           "instance":...,
           "seed":...,
           "time":...,
           "config": dictionary name -> value}

            Args:
                fn : path to "target_algo_runs.json"
        """
        timeout_list = []   # timeout runs
        censored_list = []  # censored runs
        config_list = []    # configurations (dictionaries: name->value)
        perf_list = []      # performance
        instance_list = []
        
        with open(fn) as fp:
            ct = 0
            for line in fp:
                line = line.strip("\n")
                ct += 1
                try:
                    result_dict = json.loads(line)
                    instance = result_dict["instance"]
                    # seed = result_dict["seed"]
                    status = result_dict["status"]
                    performance = result_dict["time"]
                    tmp_config = result_dict["config"]
                    misc = result_dict["misc"]
                    # Remove leading minus
                    config = dict()
                    for k in tmp_config:
                        config[k[1:]] = tmp_config[k]
                    del tmp_config
                except KeyError:
                    self.logger.warning("Could not read line in performance "
                                     "data file: %s" % line)
                    continue
                except ValueError:
                    self.logger.error("Could not read line %d: %s" % (ct, line))
                    continue

                perf, cen, timeout = self.handle_run(instance=instance,
                                                     status=status,
                                                     performance=performance,
                                                     misc=misc)
                if perf >= 0:
                    config_list.append(config)
                    perf_list.append(perf)
                    instance_list.append(instance)
                    timeout_list.append(timeout)
                    censored_list.append(cen)
                
        self.logger.debug("%s: %d configuration runs." % (fn, len(config_list)))
                    
        return config_list, perf_list, instance_list, timeout_list, \
               censored_list

    def _read_quality_json(self, fn):
        """
            parses data in generic wrapper format "target_algo_runs.json"
           {"status":...,
           "misc":...,
           "instance":...,
           "seed":...,
           "time":..., --> Performance is stored here
           "config": dictionary name -> value}

            Args:
                fn : path to "target_algo_runs.json"
        """
        config_list = []    # configurations (dictionaries: name->value)
        perf_list = []      # performance
        instance_list = []

        with open(fn) as fp:
            ct = 0
            for line in fp:
                line = line.strip("\n")
                ct += 1
                try:
                    result_dict = json.loads(line)
                    instance = result_dict["instance"]
                    # seed = result_dict["seed"]
                    status = result_dict["status"]
                    perf = float(result_dict["quality"])
                    tmp_config = result_dict["config"]
                    misc = result_dict["misc"]
                    # Remove leading minus
                    config = dict()
                    for k in tmp_config:
                        config[k[1:]] = tmp_config[k]
                    del tmp_config
                except KeyError:
                    self.logger.warning("Could not read line in performance "
                                     "data file: %s" % line)
                    continue
                except ValueError:
                    self.logger.error("Could not read line %d: %s" % (ct, line))
                    continue

                if misc is not None and misc != "":
                    self.logger.critical(misc)

                if status == "CRASHED" or status == "ABORT":
                    self.logger.critical("IGNORE CRASHED: %s, %s" % (status, misc))
                elif perf >= 0:
                    config_list.append(config)
                    perf_list.append(self.dtype(perf))
                    instance_list.append(instance)
                else:
                    self.logger.critical("IGNORE negative performance: %s" %
                                         line)
        self.logger.debug("%s: %d configuration runs." % (fn, len(config_list)))

        return config_list, perf_list, instance_list

    def handle_run(self, instance, status, performance, misc=None):
        timeout = None
        cen = None
        perf = performance

        if status in ["SAT", "UNSAT", "SUCCESS"]:
            # Regular run, everything is fine
            if performance > self.cutoff:
                logging.debug("This should have been a TIMEOUT %s, "
                              "make it one" % str((instance, status,
                                                   performance, misc)))
                perf = self.cutoff
                timeout = True
                cen = False
            else:
                timeout = False
                cen = False
        elif status == "TIMEOUT":
            # check for memory out
            if misc is not None and "memory limit was exceeded" in misc:
                self.logger.debug("IGNORE: %s" % misc)
                perf = -1
                timeout = None
                cen = None
            elif performance < self.cutoff:
                # a censored run
                cen = True
                timeout = False
                perf = performance
            elif performance >= self.cutoff:
                # a real TIMEOUT
                cen = False
                timeout = True
                perf = self.cutoff
            else:
                self.logger.debug("Don't understand this run: %s" %
                                  str((instance, status, performance,
                                       misc)))
        elif status == "CRASHED" or status == "ABORT":
            self.logger.debug("IGNORE: %s" % status)
            perf = -1
            timeout = None
            cen = None
        else:
            raise ValueError("Don't know that state: %s" % status)
        return self.dtype(perf), cen, timeout