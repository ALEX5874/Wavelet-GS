# Wavelet-GS:3D-Gaussian-Splatting-with-Wavelet-Decomposition
This is the offical repository of ACM MM 2025 "Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition"

Paper PDF: https://arxiv.org/abs/2507.12498


## ⚙️ Setup

### 0. GPU Memory Requirement
We recommend deploying Wavelet-GS on a GPU with **VRAM ≥ 24GB** for stable training and rendering.

---

### 1. Clone Repository
```
git clone https://github.com/ALEX5874/Wavelet-GS.git  
cd Wavelet-GS
```
---

### 2. Environment Setup

The environment configuration follows **Scaffold-GS**.

Please refer to the official Scaffold-GS repository for detailed installation instructions:

https://github.com/city-super/Scaffold-GS

After completing the Scaffold-GS environment setup, the same environment can be used for Wavelet-GS with slightly adjustment.

---

## 💫 Usage

### Training

The training pipeline is consistent with standard 3D Gaussian Splatting and Scaffold-GS workflows.

Example command (adjust dataset paths and configs as needed):
```
python train.py -m output_dir -s your_dataset_dir
```
---

### Rendering
```
python render.py
```
---

## 📢 Limitations

Wavelet-GS shares common limitations with Gaussian Splatting–based methods.

Performance may degrade in scenarios involving:
- Extremely sparse input views  
- Heavy occlusions  
- Highly reflective or transparent surfaces  

Further improvements may require integration with stronger geometric priors or learned visibility modeling.

---

## 🤗 Related Works

Including but not limited to:

3D Gaussian Splatting, Scaffold-GS,  Octree-GS,
Wavelet-based representations, Neural Radiance Fields (NeRF),  
Multi-resolution 3D reconstruction.

---

## 📜 Citation

If you find this work useful, please consider citing:

@inproceedings{zhao2025wavelet,  
  title={Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition},  
  author={Zhao, Beizhen and Zhou, Yifan and Yu, Sicheng and Wang, Zijian and Wang, Hao},  
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},  
  pages={8616--8625},  
  year={2025}  
}
