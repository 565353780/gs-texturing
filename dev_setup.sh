cd ..
git clone --recursive https://github.com/graphdeco-inria/gs-texturing.git gstex
git clone https://github.com/facebookresearch/pytorch3d.git

pip install plyfile tqdm

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

cd pytorch3d
python setup.py install

cd ../gstex/submodules/simple-knn
python setup.py install

cd ../diff-gaussian-rasterization-texture
python setup.py install

cd ..
python install ./graphdecoviewer
