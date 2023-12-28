#!/bin/bash
import glob
import shutil

import yaml

models = []

personas = glob.glob('personas/*.yaml')
for p in personas:
    with open(p) as f:
        y = yaml.safe_load(f)
        models.append(y['model'])

dirs = glob.glob('models/*')
for d in dirs:
    d = d.replace('models/', '')
    if (d not in models) and ('.yaml' not in d) and ('.ods' not in d) and (d != 'obsolete'):
        print(f"mv models/{d} obsolete_models/{d}")
