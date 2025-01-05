#!/bin/bash
pip3 install matplotlib scipy tqdm joblib scikit-learn torch kaolin trimesh gdown mesh-to-sdf opencv-python-headless pyrender smplx loguru albumentations pyyaml yacs scikit-image pillow pytorch-lightning numba filterpy flatten-dict jpeg4py chumpy torchgeometry torchvision timm==0.6.7
pip3 install open3d

pip3 install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
pip3 install git+https://github.com/mkocabas/yolov3-pytorch.git
pip3 install git+https://github.com/mkocabas/multi-person-tracker.git
pip3 install git+https://github.com/giacaglia/pytube.git
pip3 install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17

mkdir external && cd external

git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2_repo
pip3 install -e detectron2_repo

mkdir pare && cd pare
git clone https://github.com/mkocabas/PARE.git .
find pare/models/backbone/ -type f -name "*.py" -exec sed -i 's/torchvision\.models\.utils/torch.hub/g' {} +
sed -i "s|data/|external/pare/data/|g" external/pare/pare/core/config.py
perl -pi -e 's/\bv\.cpu\(\)\.numpy\(\)/v.cpu().float().numpy()/g' external/pare/pare/core/tester.py 
gdown 1qIq0CBBj-O6wVc9nJXG-JDEtWPzRQ4KC
unzip pare-github-data.zip

cd ..
git clone https://github.com/daniilidis-group/neural_renderer.git
find neural_renderer/ -type f -exec sed -i 's/AT_CHECK/TORCH_CHECK/g' {} +
cd neural_renderer
pip3 install -e .

cd ..
git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git
cd ChamferDistancePytorch
git checkout 719b0f1ca5ba370616cb837c03ab88d9a88173ff
python setup.py install

cd ..
pip3 install "numpy<2.0.0"
pip3 install git+https://github.com/MPI-IS/mesh.git

cd ..
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.17.0 # optional
pip3 install -e .
