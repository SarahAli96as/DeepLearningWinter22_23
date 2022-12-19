#!/bin/bash

###
# CS236781: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="part5_experiments"
MAIL_USER="majdmakhoul5@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw2

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L2_K32  -K 32 -L 2  -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L4_K32  -K 32 -L 4  -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L8_K32  -K 32 -L 8  -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L16_K32 -K 32 -L 16 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L2_K64  -K 64 -L 2  -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L4_K64  -K 64 -L 4  -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L8_K64  -K 64 -L 8  -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L16_K64 -K 64 -L 16 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5


srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L2_K32  -K 32  -L 2 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L2_K64  -K 64  -L 2 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L2_K128 -K 128 -L 2 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L4_K32  -K 32  -L 4 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L4_K64  -K 64  -L 4 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L4_K128 -K 128 -L 4 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L8_K32  -K 32  -L 8 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L8_K64  -K 64  -L 8 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L8_K128 -K 128 -L 8 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5


srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3_L2_K64-128 -K 64 128 -L 2 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3_L3_K64-128 -K 64 128 -L 3 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3_L4_K64-128 -K 64 128 -L 4 -P 2 -H 1024 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4_L8_K32  -K 32 -L 8  -P 2 -M resnet -H 2048 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4_L16_K32 -K 32 -L 16 -P 2 -M resnet -H 2048 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4_L32_K32 -K 32 -L 32 -P 2 -M resnet -H 2048 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4_L2_K64-128 -K 64 128 -L 2 -P 2 -M resnet -H 2048 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4_L4_K64-128 -K 64 128 -L 4 -P 2 -M resnet -H 2048 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4_L8_K64-128 -K 64 128 -L 8 -P 2 -M resnet -H 2048 --batches 250 --reg 5e-4 --lr 0.01 --bs-test 50 --early-stopping 5


echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

