_base_ = ['./slowfast_r18_4x16x1_256e_kinetics400_rgb.py']

#**** 输入clip32帧，tau=4 => 采样出8帧给slow分支****#
#**** slow分支 8帧，fast分支 32帧 ****#
model = dict(
    backbone=dict(
        resample_rate=4,  #! tau 用于slow分支的采样
        speed_ratio=4,  #! alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(fusion_kernel=5)))

work_dir = './work_dirs/slowfast_r18_3d_8x8x1_256e_kinetics400_rgb'
