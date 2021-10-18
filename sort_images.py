import yaml
import argparse
import os

parser = argparse.ArgumentParser(description='Sort images according to whether they can be used for calibration')
parser.add_argument('--input', dest='input', help='Input file', default='sorted_files/meta.yaml')
args = parser.parse_args()
with open(os.path.join(args.input), 'w') as f:
    outputs = yaml.load(f)
print("copying the files")
# don't modify filesystem until everything else has completed
# get the directory of the input file
input_dir = os.path.dirname(args.input)
for k,v in outputs.items():
    if not os.path.exists(path.join(input_dir,k)):
        os.makedirs(path.join(input_dir,k))
    for fpath in v:
        fname = os.path.basename(fpath)
        copy2(fpath, os.path.join(input_dir, k, fname))