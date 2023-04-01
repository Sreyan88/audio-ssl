#! /bin/bash
#SBATCH --nodes=1
#SBATCH --partition=nltmp
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=6-23:00:00
#SBATCH --error=job.%J.err
##SBATCH --output=job.%J.out
#cd $SLURM_SUBMIT_DIR
#cd /nlsasfs/home/sysadmin/nazgul/gpu-burn-master
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
#srun ./gpu_burn -tc -d 3600 #
#srun /bin/hostname
eval "$(conda shell.bash hook)"
conda activate moco_work
srun python /nlsasfs/home/nltm-pilot/ashishs/DeloresM/upstream_Delores-moco/train_moco.py \
    --input /nlsasfs/home/nltm-pilot/ashishs/pretrain_shuffled_final.csv \
    --batch-size 1024 \
    --save-path /nlsasfs/home/nltm-pilot/ashishs/DeloresM/upstream_Delores-moco/checkpoint_byol_exp_1_1024_cc_changed_128_cluster_sim_2048




