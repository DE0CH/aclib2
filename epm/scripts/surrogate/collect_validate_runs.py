import argparse
import glob
import json
import logging
import os

from epm.reader import output_parser

logger = logging.getLogger(__file__)

def main():
    parser = argparse.ArgumentParser(description='Collect runs from validate '
                                                 'folders and output a json file')
    parser.add_argument('val_folder', metavar='VALDIR', nargs='+',
                        help='Path to validate folder')
    parser.add_argument('--out', dest='out', default=None, required=True,
                        help='Where to save file?')
    parser.add_argument('--cutoff', dest='cutoff', default=None,
                        type=float, help='What was the cutoff? '
                                         'If None assume quality')

    logging.basicConfig(level=logging.INFO)
    args, unknown = parser.parse_known_args()

    if args.cutoff is None:
        logging.critical("Assuming QUALITY")

    assert not os.path.isfile(args.out), "%s already exists" % args.out

    for v_folder in args.val_folder:
            try:
                val_obj_fl = glob.glob(os.path.join(v_folder, "validationObjectiveMatrix-*.csv"))[0]
                val_str_fl = glob.glob(os.path.join(v_folder, "validationCallStrings-*.csv"))[0]
            except IndexError:
                logging.critical("Can't find files")
                raise

            assert os.path.exists(val_obj_fl)
            assert os.path.exists(val_str_fl), val_str_fl

            # We first read all perfs and call_str
            perfs = output_parser.read_validationObjectiveMatrix_file(val_obj_fl)
            call_str = output_parser.read_validationCallStrings_file(val_str_fl)
            ct = 0

            with open(args.out, 'wb') as fh:
                for inst in perfs.keys():
                    for idx, config in enumerate(call_str):
                        assert len(call_str) == len(perfs[inst])
                        tmp_dict = dict()

                        # Set performance
                        tmp_dict['time'] = float(perfs[inst][idx])

                        # Set status
                        if args.cutoff is None:
                            tmp_dict['status'] = 'SAT'
                            tmp_dict['quality'] = tmp_dict['time']
                            del tmp_dict['time']
                        else:
                            if tmp_dict['time'] >= args.cutoff:
                                tmp_dict['status'] = 'TIMEOUT'
                            else:
                                tmp_dict['status'] = 'SAT'

                        # Set config
                        t_dict = dict()
                        for k in call_str[idx]:
                            t_dict['-%s' % k] = call_str[idx][k]
                        tmp_dict['config'] = t_dict

                        # Set remaining keys
                        tmp_dict['misc'] = ""
                        tmp_dict['seed'] = 23
                        tmp_dict['instance'] = inst

                        json.dump(tmp_dict, fh, sort_keys=False)
                        fh.write("\n")
                        ct += 1
            logging.info("Written %d to %s" % (ct, args.out))

if __name__ == "__main__":
    main()