source /home/chenkangyang/anaconda3/bin/activate
conda activate mm
export CUDA_VISIBLE_DEVICES='1,3,6,7'

CONFIG_FILE=configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint.py
GPU_NUM=4
WORK_DIR=work_dirs/slowonly_r50_u48_240e_gym_keypoint
SEED=0

# 单卡训练
# python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint.py \
#     --work-dir work_dirs/slowonly_r50_u48_240e_gym_keypoint \
#     --validate --seed 0 --deterministic \
#     --gpu-ids 1

# 多卡训练
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} \
    --work-dir ${WORK_DIR} \
    --validate --seed ${SEED} --deterministic

# --validate (strongly recommended): Perform evaluation at every k (default value is 5, which can be modified by changing the interval value in evaluation dict in each config file) epochs during the training.
# --test-last: Test the final checkpoint when training is over, save the prediction to ${WORK_DIR}/last_pred.pkl.
# --test-best: Test the best checkpoint when training is over, save the prediction to ${WORK_DIR}/best_pred.pkl.
# --work-dir ${WORK_DIR}: Override the working directory specified in the config file.
# --resume-from ${CHECKPOINT_FILE}: Resume from a previous checkpoint file.
# --gpus ${GPU_NUM}: Number of gpus to use, which is only applicable to non-distributed training.
# --gpu-ids ${GPU_IDS}: IDs of gpus to use, which is only applicable to non-distributed training.
# --seed ${SEED}: Seed id for random state in python, numpy and pytorch to generate random numbers.
# --deterministic: If specified, it will set deterministic options for CUDNN backend.
# JOB_LAUNCHER: Items for distributed job initialization launcher. Allowed choices are none, pytorch, slurm, mpi. Especially, if set to none, it will test in a non-distributed mode.
# LOCAL_RANK: ID for local rank. If not specified, it will be set to 0.