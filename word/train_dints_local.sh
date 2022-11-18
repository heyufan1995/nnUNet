export nnUNet_raw_data_base='/home/yufan/Data/'
export RESULTS_FOLDER='/home/yufan/Data/nnUNet_raw_data/result/'
export nnUNet_preprocessed='/home/yufan/Data/nnUNet_raw_data/preprocess/'
cp /home/yufan/Projects/nnUNet/word/nnUNetTrainerV2_DiNTS.py  /home/yufan/Projects/nnUNet/nnunet/training/network_training/nnUNetTrainerV2.py
export CUDA_VISIBLE_DEVICES=0,1 
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=1 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_DDP Task001_Word 0 --dbs
# nnUNet_train 3d_fullres nnUNetTrainer Task001_Word 0 --npz
