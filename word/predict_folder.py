from nnunet.inference.predict import predict_from_folder 

predict_from_folder(model='/home/yufan/Projects/dlmed_transformers/cfswin/cvpr_legacy/nnUNet3DV1_PreTrained_Model/V1',
                    input_folder='/home/yufan/Data/nnUNet_raw_data/Task001_Word/imagesTs/',
                    output_folder='./',
                    folds=None, save_npz=True, num_threads_preprocessing=8,
                    num_threads_nifti_save=2, lowres_segmentations=None, part_id=0, num_parts=1, tta=False)