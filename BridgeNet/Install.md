conda create -n eemtl python=3.8
conda activate eemtl

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -U openmim
mim install mmsegmentation==0.30.0 mmcv==1.7.0

pip install tqdm Pillow easydict pyyaml imageio scikit-image tensorboard wandb
pip install opencv-python==4.5.4.60 setuptools==59.5.0
pip install timm==0.5.4 einops==0.4.1