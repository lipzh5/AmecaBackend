
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
```