#!/usr/bin/env python

import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

job_directory = "%s/jobs" %os.getcwd()
# Make top level directories
mkdir_p(job_directory)

jn = 'weatherbench_cnn_pgpu_test_2D'
job_file = os.path.join(job_directory,"%s.job" %jn)
with open(job_file, 'w') as fh:
    fh.writelines("#!/bin/tcsh\n")
    fh.writelines("#SBATCH --job-name=%s.job\n" % jn)
    fh.writelines("#SBATCH --partition=pGPU")
    fh.writelines("#SBATCH --nodes=1")
    fh.writelines("#SBATCH --nodeslist=g002")
    fh.writelines("#SBATCH --time=12:00:00\n")
    fh.writelines("#SBATCH --output=jobs/%s.out\n" % jn)
    fh.writelines("#SBATCH --error=jobs/%s.err\n" % jn)
    fh.writelines("#SBATCH --mail-type=END\n")
    fh.writelines("#SBATCH --mail-user=nonnenma@hzg.de\n")
    fh.writelines("#SBATCH --account=nonnenma\n")        
    fh.writelines("hostname\n")
    fh.writelines("nvidia-smi\n")
    fh.writelines("cd /gpfs/home/nonnenma/projects/seasonal_forecasting/code/weatherbench/\n")
    fh.writelines("python run_cnn_pgu_pytorch.py\n")


os.system("sbatch %s" %job_file)