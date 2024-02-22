#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J social-aware-perception
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request GB of system-memory
#BSUB -R "rusage[mem=8GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u dimara@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err
# -- end of LSF options --

# Load the cuda module
#module load numpy/1.21.1-python-3.8.11-openblas-0.3.17
#module load cuda/11.4


#Activate virtual env
unset PYTHONPATH
unset PYTHONHOME
source ~/miniconda3/bin/activate
conda activate social-aware-perception

#python test_script.py
python trainer_humann_3dloc.py --config configs/config_full_data_transformer.yaml
