# Setup
- [create conda environment]()
```
conda create -n zmqbackend python=3.9
conda activate zmqbackend
```
- [install necessary libs]()
```
pip install pyzmq
pip install torch
# conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers
pip install aiohttp 
# install hiera from source
git clone https://github.com/facebookresearch/hiera.git
cd hiera
python setup.py build develop
# action generation
pip install openai
# face recognition-insightface
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
# pip install onnxruntime-gpu  # cuda11.8,  or onnxruntime for cpu only inference, ref: https://onnxruntime.ai/docs/install/
pip install insightface
```
