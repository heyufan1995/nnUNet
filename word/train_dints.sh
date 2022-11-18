pip install monai
pip install einops
export nnUNet_raw_data_base='/data/words/'
export RESULTS_FOLDER='/workspace/nnUNet/word/result'
export nnUNet_preprocessed='/data/words/preprocess/'
cp nnUNetTrainer_DiNTS.py  ../nnunet/training/network_training/nnUNetTrainer.py
# export CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch --master_port=1234 --nproc_per_node=Y run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_DDP TASK_NAME_OR_ID FOLD --dbs
nnUNet_train 3d_fullres nnUNetTrainer Task001_Word 0 --npz
