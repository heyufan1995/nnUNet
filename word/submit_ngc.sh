

TASK='words'
# ID=110269
ID=217257
FOLD=0
GPU=16g
NUMGPU=2
INSTANCE=dgx1v.$GPU.$NUMGPU.norm
# TASK='Task05_Prostate'
# ID=68113
# FOLD=4
# GPU=16g
# NUMGPU=1
# INSTANCE=dgx1v.$GPU.$NUMGPU.norm 

ngc batch run \
--team "dlmed" \
--name "ml-model.nnunet" \
--preempt RESUMABLE \
--min-timeslice 0s \
--total-runtime 0s \
--image "nvcr.io/nvidia/pytorch:22.08-py3" \
--ace nv-us-west-2 \
--instance $INSTANCE \
--result /results \
--datasetid $ID:/data/$TASK/ \
--workspace yufan_nas:/workspace \
--port 6006 \
--commandline "cd /workspace/nnUNet/word;bash train_dints.sh" 


