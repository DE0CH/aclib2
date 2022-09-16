#!/usr/bin/env python3

import os
import re

things = {}
with open('scenario.txt') as f:
  key = ''
  value = ''
  watch = False
  for line in f:
    if line.strip().startswith('arg'):
      if re.match(r'^arg\s*=(.*)$', line).group(1).strip().startswith('--'):
        key = re.match(r'^arg\s*=(.*)$', line).group(1).strip()
        watch = True
      elif watch:
        value = re.match(r'^arg\s*=(.*)$', line).group(1).strip()
        things[key] = value
        watch = False

things = {your_key: things[your_key] for your_key in ['--pyrfr_wrapper', '--pyrfr_model', '--config_space', '--inst_feat_dict']}

with open('extra_args.txt', 'w') as f:
  for key in things:
    f.write(f'{key} {things[key]} ')

print(things)