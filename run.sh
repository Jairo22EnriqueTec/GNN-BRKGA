#!/bin/bash

# sbatch options (can be passed on command-line)
# cfr. sbatch(1)
#SBATCH --job-name=extractfeats
#SBATCH --time=05-00:00:00
#SBATCH --mem=10G
#SBATCH --nodes=5

#SBATCH --output=feats%J.log
#SBATCH --error=feats-extractfeats-%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1651580@uab.cat

# This is ignored, line must begin exactly with '#SBATCH '
## #SBATCH --cpus-per-task=10

# You must carefully match tasks, cpus, nodes,
#  and cpus-per-task for your job. See docs.
## #SBATCH --ntasks=1
## #SBATCH --cpus-per-task=10

# Load all needed modules
spack load python@3.8.11
spack load py-networkx
spack load py-numpy@1.21.3

# change to project/software directory
cd /mnt/beegfs/iiia/jairo_ramirez/socialnetworks_feats/

# run the program
python savefeats.py -p "./instances/txt/" -ps "./feats/"