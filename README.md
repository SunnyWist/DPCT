# DPCT
## Setup (Anaconda)
- {environmental name} is the name of the environment you want to create.
- You should have a CUDA GPU and drivers installed.
1. `conda create -n {environmental name} python=3.8` # Python 3.8 (not <=3.7 or >=3.9) is required
2. `conda activate {environmental name}`
3. `python3 -m pip install -r requirements.txt`
4. `conda install -c conda-forge libstdcxx-ng`
5. `python3 -m pip install nvidia-pyindex`
6. `python3 -m pip install nvidia-tensorflow[horovod]`
7. `python3 -m pip install nvidia-tensorboard`
8. `bash cpp_setup.sh`

## How to run
- `bash generate_envs.sh` # Generate the environments
- `bash train.sh` # Train the model
- `bash simulate.sh` # Simulate the model

## How to change the parameters
- Change the files in `configs/` directory

## References
- [PRIMAL2 - marmotlab](https://github.com/marmotlab/PRIMAL2)
