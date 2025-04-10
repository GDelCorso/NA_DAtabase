import argparse
import sys
from multiprocessing import get_start_method
from datetime import datetime

sys.path.append('NA_DAtabase_Sampler')

import NA_DAtabase_Sampler as NA_DA_S

parser = argparse.ArgumentParser(
    prog='generate-image-cli',
    usage='%(prog)s [options]',
    description='This command generate images from a dataset path. The path must contain the file combined_dataframe.csv',
    )

parser.add_argument('-p', '--path',help='path of the dataset', required=True)

try:
    options = parser.parse_args()
except:
    parser.print_help()
    sys.exit(1)


start_time = datetime.now() 

print("Preparing data from %s "% options.path)

enable_multiprocess = True

if get_start_method() == 'fork':
	enable_fork = True
else:
	enable_fork = False
	enable_multiprocess = False
		

NA_DA_S.random_sampler(
	give_full_path=options.path, 
	gui=None, 
	enable_multiprocess=enable_multiprocess, 
	enable_fork=enable_fork).auto_process()
		
time_elapsed = datetime.now() - start_time 

print("Done in %s" % str(time_elapsed))

