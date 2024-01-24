### Installation guide, 12.12.

* Conda Enviroment
```bash
  conda create -n deca python==3.9
  ```

* Cuba Requirements
```bash
  conda install -c "nvidia/label/cuda-11.6.1" cuda
  ```
* Required packages
```bash
    pip install -r requirements.txt
  ```
* Pytorch Install guide
```bash
    conda create -n pytorch3d python=3.9
    conda activate pytorch3d
    conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub
    conda install pytorch3d -c pytorch3d
  ```

* Numpy version will change while isntalling pytorch3d
```bash
    conda install numpy==1.23.5
```

* Testing via pytroch3d Rasterizer
```bash
    CUDA_HOME=$CONDA_PREFIX python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True --rasterizer_type pytorch3d
```
* Testing own image
```bash
    CUDA_HOME=$CONDA_PREFIX python demos/demo_reconstruct.py -i TestSamples/Brenken_Rudolf --saveDepth True -s TestSamples/Brenken_Rudolf/results --saveObj True --rasterizer_type pytorch3d
```

