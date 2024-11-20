# Setup
- [create conda environment]()
```
conda create -n zmqbackend python=3.9
conda activate zmqbackend
```
- [install necessary libs]()
```
git clone https://github.com/lipzh5/HumanoidBackend.git
cd HumanoidBackend
pip install -r requirements.txt

# install hiera from source
cd ..
git clone https://github.com/facebookresearch/hiera.git
cd hiera
python setup.py build develop
```
