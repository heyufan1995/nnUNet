cd /workspace/
git clone https://github.com/heyufan1995/nnUNet.git
cd nnUNet
pip install -e .
pip install monai
export nnUNet_raw_data_base='/data/words/'
export RESULTS_FOLDER='/workspace/nnUNet/result'
export nnUNet_preprocessed='/data/words/preprocess/'
echo 'renaming word datasets'
python runword.py
echo 'run preprocessing'
nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity
echo 'move split to preprocessed folder'
cp 'split_final.pkl' "$nnUNet_preprocessed/Task001_Word/split_final.pkl"
echo 'move dints trainer'
cp nnUNetTrainer_DiNTS.py  ../nnunet/training/network_training/nnUNetTrainer.py
nnUNet_train 3d_fullres nnUNetTrainer Task001_Word 0 --npz
