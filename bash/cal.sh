export CUDA_VISIBLE_DEVICES=1

#### caltech101 16
SAVE_DIR=/export/livia/home/vision/Mnoori/projects/few/FewShot-CLIP-Strong-Baseline/all_save/all_temps/caltech101_16
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "itap of a {}" --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a bad photo of the {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a origami {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a photo of the large {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a {} in a video game." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "art of the {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a photo of the small {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 16 tasks 10










#### caltech101 4
SAVE_DIR=/export/livia/home/vision/Mnoori/projects/few/FewShot-CLIP-Strong-Baseline/all_save/all_temps/caltech101_4
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 4 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "itap of a {}" --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 4 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a bad photo of the {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 4 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a origami {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 4 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a photo of the large {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 4 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a {} in a video game." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 4 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "art of the {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 4 tasks 10
python main.py --base_config configs/base.yaml --dataset_config configs/caltech101.yaml --template "a photo of the small {}." --opt root_path /projets/Mnoori/data/few/ output_dir $SAVE_DIR method LinearProbe_P2 shots 4 tasks 10





