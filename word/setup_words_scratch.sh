cd /workspace/
git clone https://github.com/heyufan1995/nnUNet.git
cd nnUNet
pip install -e .
pip install monai
pip install einops
export nnUNet_raw_data_base='/data/words/'
export RESULTS_FOLDER='/workspace/nnUNet/word/result'
export nnUNet_preprocessed='/data/words/preprocess/'
cd word/
echo 'renaming word datasets'
python runword.py
mv dataset.json $nnUNet_raw_data_base/nnUNet_raw_data/Task001_Word/
echo 'run preprocessing'
nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity
echo 'move split to preprocessed folder'
cp 'splits_final.pkl' "$nnUNet_preprocessed/Task001_Word/splits_final.pkl"
echo 'move dints trainer'
cp /workspace/nnUNet/word/nnUNetTrainer_DiNTS.py  /workspace/nnUNet/nnunet/training/network_training/nnUNetTrainer.py
cp /workspace/nnUNet/word/nnUNetTrainer_SwinUNETR.py  /workspace/nnUNet/nnunet/training/network_training/nnUNetTrainer.py
nnUNet_train 3d_fullres nnUNetTrainer Task001_Word 0 --npz
