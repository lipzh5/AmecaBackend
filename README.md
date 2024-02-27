
## Setup
- [create conda environment]()
```
conda create -n zmqbackend python=3.9
conda activate zmqbackend
```
- [install necessary libs]()
```
pip install pyzmq
pip install torch
pip install numpy
pip install Pillow
# install hiera from source
git clone https://github.com/facebookresearch/hiera.git
cd hiera
python setup.py build develop
# action generation
pip install openai
# face recognition-insightface
pip install onnxruntime-gpu  # or onnxruntime for cpu only inference
pip install insightface
```