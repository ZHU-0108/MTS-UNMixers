<div align="center">
<h1>MTS-UNMixers</h1>
<h3>Multivariate Time Series Forecasting via Channel-Time Dual Unmixing</h3>

[Xuanbing Zhu](https://github.com/ZHU-0108/FCNet-main)<sup>1</sup> , [Dunbin Shen](https://scholar.google.com/citations?user=DH4VSLMAAAAJ&hl=zh-CN)<sup>1</sup> , Zhongwen Rao<sup>2</sup> , Huiyi Ma<sup>1</sup>, Yingguang Hao<sup>1</sup>, [Hongyu Wang](http://faculty.dlut.edu.cn/MMCL_WHY/zh_CN/)<sup>1</sup>, [Xiaorui Ma](https://scholar.google.com/citations?hl=zh-CN&user=bM2EAnMAAAAJ)<sup>1 :email:</sup>

<sup>1</sup>  Dalian University of Technology, <sup>2</sup>  Huawei Noah’s Ark Lab

(<sup>:email:</sup>) Corresponding author.

ArXiv Preprint ([arXiv 2411.17770](https://arxiv.org/pdf/2411.17770))


</div>


#



### News
* **` November 26th, 2024`:** We released our codes and models.️


## Abstract
Multivariate time series data provide a robust framework for future predictions by leveraging information across multiple dimensions, ensuring broad applicability in practical scenarios. However, their high dimensionality and mixing patterns pose significant challenges in establishing an interpretable and explicit mapping between historical and future series, as well as extracting long-range feature dependencies. To address these challenges, we propose a channel-time dual unmixing network for multivariate time series forecasting (named MTS-UNMixer), which decomposes the entire series into critical bases and coefficients across both the time and channel dimensions. This approach establishes a robust sharing mechanism between historical and future series, enabling accurate representation and enhancing physical interpretability. Specifically, MTS-UNMixers represent sequences over time as a mixture of multiple trends and cycles, with the time-correlated representation coefficients shared across both historical and future time periods. In contrast, sequence over channels can be decomposed into multiple tick-wise bases, which characterize the channel correlations and are shared across the whole series. To estimate the shared time-dependent coefficients, a vanilla Mamba network is employed, leveraging its alignment with directional causality. Conversely, a bidirectional Mamba network is utilized to model the shared channel-correlated bases, accommodating noncausal relationships. Experimental results show that MTS-UNMixers significantly outperform existing methods on multiple benchmark datasets. The code is available at https://github.com/ZHU-0108/MTS-UNMixers.
## Overview


## Installation
- CUDA 11.7
  - Make sure `/usr/local/cuda-11.7` exists. If not, you can install it from NVIDIA DEVELOPER ([CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)). For example, Ubuntu 18.04 x86_64
    - `wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run`
    - `sudo sh cuda_11.7.1_515.65.01_linux.run`
  
    Note that if you use Ubuntu 24.04, perhaps the gcc version is too high to install CUDA 11.7. So you should degrade the gcc version at first like this:
    - `sudo apt install gcc-9 g++-9`
    - `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 `
    - `gcc -v`
  - See `nvcc -V` is `cuda_11.7`. If not, you should modify the `.bashrc` like this:
    - `vim ~/.bashrc` -> `i`, and add the following to the end
    
      export CUDA_HOME=/usr/local/cuda-11.7
    
      export PATH=$PATH:/usr/bin:/bin
    
      export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
    
      export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    - `esc`->`:wq`->`source ~/.bashrc`, and see `nvcc -V` is `cuda_11.7`.
- Python 3.10.x
  - `conda create -n htd-mamba python=3.10`
  - `conda activate htd-mamba`

- Torch 2.0.1
  - `conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia` or
  - `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117`

- Requirements: requirements.txt
  - `cd /home/your_path/HTD-Mamba-main`
  - `pip install -r requirements.txt`

- Install ``selective_scan_cuda``
  - `cd /home/your_path/HTD-Mamba-main`
  - `pip install .`
  
- Install ``causal_conv1d``
  - `pip install --upgrade pip`
  - `pip install causal_conv1d>=1.1.0`
 

## Training
- To train `MTS-UNMixers`, change the state to `train`.

## Acknowledgement
This project is based on `Mamba` ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)). We thank the authors for their promising studies.

## Citation
If you find `MTS-UNMixers` useful in your research or applications, please consider giving us a star 🌟 and citing it using the following BibTeX entry.

```bibtex
@misc{zhu2024mtsunmixersmultivariatetimeseries,
      title={MTS-UNMixers: Multivariate Time Series Forecasting via Channel-Time Dual Unmixing}, 
      author={Xuanbing Zhu and Dunbin Shen and Zhongwen Rao and Huiyi Ma and Yingguang Hao and Hongyu Wang},
      year={2024},
      eprint={2411.17770},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.17770}, 
}
```
