import pandas as pd
import argparse
import loc_utils as lut
from standards import RAWix

parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to file to view')
args = parser.parse_args()

data = lut.unpickle(args.path)['main']
r = RAWix()

df = pd.DataFrame(data, columns=r.cols)

df.to_csv('output.csv')