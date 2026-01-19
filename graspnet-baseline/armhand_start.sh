#unset PYTHONPATH
#sudo chmod 777 /dev/ttyACM*
CUDA_VISIBLE_DEVICES=0 python arm_hand_grasp.py --checkpoint_path logs/log_kn/checkpoint.tar