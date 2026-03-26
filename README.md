<div align=center>
<h1>NodePFN: Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors</h1>

![GitHub Repo stars](https://img.shields.io/github/stars/jeongwhanchoi/NodePFN) ![Twitter Follow](https://img.shields.io/twitter/follow/jeongwhan_choi?style=social)

<div>
    <a href="https://www.jeongwhanchoi.com" target="_blank"><b>Jeongwhan Choi</b></a><sup>1*</sup>,
    Jongwoo Kim<sup>1*</sup>, Woosung Kim<sup>1</sup>,
    <a href="https://sites.google.com/view/noseong" target="_blank">Noseong Park</a><sup>1</sup>,
    <div>
    <sup>1</sup>KAIST
    </div>
</div>
</div>


---

Official implementation of **NodePFN**, accepted at **ICLR 2026**.
> **TL;DR:** A single pre-trained model. No graph-specific training. Universal node classification.

---
## Installation

**Requirements:** Python 3.11, CUDA 12.1, NVIDIA GPU (tested on RTX 6000 / A6000)
 
- **Option 1**: `requirements.txt`
 
```bash
git clone https://github.com/jeongwhanchoi/NodePFN.git
cd NodePFN
pip install -r requirements.txt
```
 
- **Option 2**: Manual installation of core dependencies
 
```bash
git clone https://github.com/jeongwhanchoi/NodePFN.git
cd NodePFN
 
pip install torch==1.12.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install numpy==1.26.4 networkx==3.3 scikit-learn==1.4.0
# Note: The above packages list only includes the core dependencies. Additional packages may be required for pre-training or specific datasets, but the above should suffice for inference on standard benchmarks.
```

---
## Pre-training
Pre-train the NodePFN model on synthetic graph priors. **This step is required only once** — the resulting checkpoint can then be used for inference on arbitrary graphs.

```bash
python nodepfn/main.py --model_name {your_model_name}
```

> A pre-trained checkpoint is also available for download. See [Releases](models_ckpts/nodepfn/checkpoint_epoch_30.ckpt).


## Inference

No fine-tuning needed. Run inference directly with the pre-trained model.
 
All per-dataset commands are in [`run.sh`](run.sh)
 
<details>
<summary>Example: Cora</summary>
 
```bash
# Cora
python nodepfn/node_classification.py \
    --dataset cora \
    --base_model_path=models_ckpts/nodepfn \
    --dim_reduction tsvd --n_components 15 \
    --runs=5 --smoothing_steps 4
```
 
</details>

---

## Limitations
 
- Supports up to **20 classes**; features beyond model capacity are reduced via truncated SVD.
- Attention complexity is **O(N²d)** — large-scale graphs are not yet supported.
- One-time pre-training cost: ~6 GPU hours on a single NVIDIA RTX A6000.

> [!IMPORTANT]
> **A note on reproducibility.**
> Exact reproduction of the paper's numbers is not guaranteed due to several sources of non-determinism:
> - **Pre-training**: synthetic graphs are sampled stochastically at each step, so re-training from scratch will not yield an identical checkpoint.
> - **Inference**: results may vary across GPU architectures, CUDA versions, and library versions.
>
> To minimize variance, we recommend using the **released checkpoint** from [Releases](models_ckpts/nodepfn/checkpoint_epoch_30.ckpt).

---

## Citation

If you use NodePFN in your research, please cite:

```bibtex
@inproceedings{choi2026nodepfn,
  title     = {Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors},
  author    = {Choi, Jeongwhan and Kim, Jongwoo and Kang, Woosung and Park, Noseong},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=NodePFN}
}
```

---

## Related Github Repository
Our implementation is built on top of the [TabPFN-v1](https://github.com/PriorLabs/TabPFN/tree/tabpfn_v1/) framework.