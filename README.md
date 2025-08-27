# Efficient Robotic Policy Learning via Latent Space Backward Planning
[[Website](https://lbp-authors.github.io/)]  [[Paper](https://www.arxiv.org/abs/2505.06861)]

🔥 **LBP has been accepted by ICML 2025**

## Introduction

We propose a Latent Space Backward Planning scheme (LBP), which balances efficiency, accuracy and sufficient future guidance. LBP begins by grounding the task into final latent goals, followed by recursively predicting intermediate subgoals closer to the current state. The grounded final goal enables backward subgoal planning to always remain aware of task completion, facilitating ontask prediction along the entire planning horizon. The subgoal-conditioned policy incorporates a learnable token to summarize the subgoal sequences and determines how each subgoal guides action extraction. Through extensive simulation and real-robot long-horizon experiments, we show that LBP outperforms existing fine-grained and forward planning methods, achieving SOTA performance.

<p align="center"> 
	<img src="assets/images/LBP_intro.jpg"width="100%"> 
</p>


## Quick Start

### Installation

1. Clone this repository and create an environment.
```bash
git clone git@github.com:Dstate/LBP.git
conda create -n lbp python=3.8 -y
conda activate lbp

```

2. Set up [DecisionNCE](https://github.com/2toinf/DecisionNCE).
```bash
git clone https://github.com/2toinf/DecisionNCE.git
cd DecisionNCE
pip install -e .
cd ..
```

3. We use the checkpoint of [DecisionNCE(Robo-MUTUAL)](https://github.com/255isWhite/Robo_MUTUAL), which can be downloaded from [link](https://drive.google.com/file/d/1_bvhXUzWYWhg7bUANhDRB9Zq09wKcjB1/view?usp=drive_link).
```bash
mkdir -p ~/.cache/DecisionNCE
mv <above_downloaded_ckpt> DecisionNCE-T
mv DecisionNCE-T ~/.cache/DecisionNCE
```

4. Set up [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ..
```

5. Install other package for LBP.
```bash
cd LBP
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### Dataset Preprocessing
We follow [OpenVLA](https://github.com/openvla/openvla) to regenerate the Libero dataset into 256×256 resolution and filter out failed rollouts (by running `openvla/experiments/robot/libero/regenerate_libero_dataset.py`). 

Note that we use a different HDF5 format from the original. To convert your dataset to our format, run the following script:
```
python hdf52hdf5.py
```
An example of the our format can be found [here](https://drive.google.com/drive/folders/1cmJNChezV--pfy1UgmJfTCT3W-7MCd6Z?usp=sharing). You can also directly download our preprocessed data from the same link.

### Training and evaluation

1. Train the latent backward planner.
```bash
bash scripts/planner_libero.sh
```

2. Then train the policy.
```bash
bash scripts/lbp_ddpm-libero_10.sh
```

3. Evaluate the policy on LIBERO benchmark.
```bash
python eval_libero.py
```

## Citation
If the paper or the codebase is helpful with your research, please cite it as:
```
@inproceedings{
    liu-niu2025lbp,
    title={Efficient Robotic Policy Learning via Latent Space Backward Planning},
    author={Dongxiu Liu and Haoyi Niu and Zhihao Wang and Jinliang Zheng and Yinan Zheng and zhonghong Ou and Jianming Hu and Jianxiong Li and Xianyuan Zhan},
    booktitle={International Conference on Machine Learning},
    year={2025}
}
```

## License
All the code, model weights, and data are licensed under [MIT license](LICENSE).
