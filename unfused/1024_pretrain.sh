#! /bin/bash
#SBATCH --nodes=1
#SBATCH --partition=nltmp
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=4-23:00:00
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
#source ~/miniconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate moco_work
srun python /nlsasfs/home/nltm-pilot/ashishs/DeloresM/upstream_SSSD_wm/train_moco.py \
    --input /nlsasfs/home/nltm-pilot/ashishs/DECAR/libri_100_new.csv \
    --batch-size 256 \
    --save-path /nlsasfs/home/nltm-pilot/ashishs/DeloresM/upstream_SSSD_wm/checkpoint_libri_100_new_arch_linear_projector_lr_007_cln_256_wm



