#### caltech101 16
SAVE_DIR=/export/livia/home/vision/Mnoori/projects/few/FewShot-CLIP-Strong-Baseline/all_save/all_temps/caltech101_16
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template  --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10





#### caltech101 4
SAVE_DIR=/export/livia/home/vision/Mnoori/projects/few/FewShot-CLIP-Strong-Baseline/all_save/all_temps/caltech101_4



#### SUN397 16
SAVE_DIR=/export/livia/home/vision/Mnoori/projects/few/FewShot-CLIP-Strong-Baseline/all_save/all_temps/SUN397_16
python main.py --base_config configs/base.yaml --dataset_config configs/sun397.yaml --template --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10


#### SUN397 4
SAVE_DIR=/export/livia/home/vision/Mnoori/projects/few/FewShot-CLIP-Strong-Baseline/all_save/all_temps/SUN397_4
