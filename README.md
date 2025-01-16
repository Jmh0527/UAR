# UAR: Detecting AI-Generated Images Using Only Real Images

The rapid advancement of image generation technologies has led to highly realistic synthetic images, which pose a threat to public trust and security. In real-world scenarios where the generative model behind a fake image is unknown, an effective detector should be capable of identifying out-of-distribution generated images. Current methods typically focus on identifying common artifacts across different generative models. However, these methods often erase a substantial portion of image information, resulting in detection failures even when fakes are visually distinguishable. Our experiments show that autoregressive features are effective for detecting generated images, allowing us to eliminate the reliance on artifacts and preserve the image information. Building on this finding, we propose Universal Autoregressive Detection (UAR), which is trained exclusively on real images. We generate negative samples by reconstructing real images with a variational autoencoder (VAE), rather than using AI-generated images. UAR achieves an average accuracy of 94.48% across 16 generative models, surpassing the state-of-the-art (SoTA) by 2.79%. Additionally, UAR exhibits stable performance under various perturbations.

<div style="text-align: center;">
  <img src="./figure/pipeline.jpg" alt="pipeline" width="800">
</div>


## Environment Setup

To create the environment and install the necessary dependencies, use the following commands:

```bash
# Create a virtual environment named UAR with Python version 3.10.15
conda create -n UAR python==3.10.15
conda activate UAR

# Install PyTorch with CUDA 11.8 support, along with torchvision and torchaudio
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install other package
pip install opencv-python
pip install tqdm
pip install transformers==4.47.0

```

## Feature Extraction
We use **AIMv1** to extract image features. The original code can be found at [AIM GitHub repository](https://github.com/apple/ml-aim).  

We made certain modifications to simplify the environment configuration. The modified version is stored in the `UAR/aim` directory.

```bash
cd UAR

# Download the AIM model checkpoint for the backbone weights
wget -O aim_3b_5bimgs_attnprobe_backbone.pth "https://cdn-lfs-us-1.hf.co/repos/1d/1f/1d1f735a636a3cee919e7ab99cb59ab0608b7194e5f6e3569464ab1f9fb28032/8475ce4e4b2b618403d267393f4fac00f614f3bad26b8389506e7762b394260a?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27aim_3b_5bimgs_attnprobe_backbone.pth%3B+filename%3D%22aim_3b_5bimgs_attnprobe_backbone.pth%22%3B&Expires=1737077195&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNzA3NzE5NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzFkLzFmLzFkMWY3MzVhNjM2YTNjZWU5MTllN2FiOTljYjU5YWIwNjA4YjcxOTRlNWY2ZTM1Njk0NjRhYjFmOWZiMjgwMzIvODQ3NWNlNGU0YjJiNjE4NDAzZDI2NzM5M2Y0ZmFjMDBmNjE0ZjNiYWQyNmI4Mzg5NTA2ZTc3NjJiMzk0MjYwYT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=ba-u3ubj1C5bgZuUF3yiYswd9xUbc-odrm79e29mAA3sVeCBZhrQ-LnFDIVeq8scMcBOeL2TAK4AHhAQGjz5lHRpOuqF6g3TciT9g42ptQh2w8EGgVP%7EBVGCQ%7E436t5yQ0SvyxXPO%7EJQldooJvaZF-FVMGBY6esaW8GxgVHmQRzcQ6KJD9-dUUGp5fcX7NgIP-qNw95he2PZ-E1ma%7Eui-0kApum%7EL7MMqtR8H9lK6uYMXpQJwvzYxeLaetZh9V6BGjviBKQNr0QbIHB7VHxSDIvfV2p9EN9qLtz1-IyQALZYTIOJBdlXt2tcmRYB229HxkxKKN3agz0oGQvQOFNOIA__&Key-Pair-Id=K24J24Z295AEI9"

# Download the second AIM model checkpoint for the head layer weights (best layers)
wget -O aim_3b_5bimgs_attnprobe_head_best_layers.pth "https://cdn-lfs-us-1.hf.co/repos/1d/1f/1d1f735a636a3cee919e7ab99cb59ab0608b7194e5f6e3569464ab1f9fb28032/ad380e16491c30513e7bae84e7b7569272f46b3989e87011ee2574e4bc775586?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27aim_3b_5bimgs_attnprobe_head_best_layers.pth%3B+filename%3D%22aim_3b_5bimgs_attnprobe_head_best_layers.pth%22%3B&Expires=1737077208&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNzA3NzIwOH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzFkLzFmLzFkMWY3MzVhNjM2YTNjZWU5MTllN2FiOTljYjU5YWIwNjA4YjcxOTRlNWY2ZTM1Njk0NjRhYjFmOWZiMjgwMzIvYWQzODBlMTY0OTFjMzA1MTNlN2JhZTg0ZTdiNzU2OTI3MmY0NmIzOTg5ZTg3MDExZWUyNTc0ZTRiYzc3NTU4Nj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=m7S8q%7EH2ABIxtMdGyNih4ixwSrhQHZ9ARilGCVqYOdFeT4f5gW8UGrSoJ1FayJu2yjgjxu3BS0YOeuzYIP9oLEaDnzF-pp3%7EmajbiwImxVvrY2KKtX0jE89dQ3SqjFHrEzeVDrIDt-uEOIdvXFh8m%7EEGyTAb2-OqZ8Gl1pJzWOQLnvT4m-ukU7i9RnH0Ej5OVpBcoyfULgZrJ7a3J%7ETxHstUlWmOwJreAFjRztzMWn-V88VwJD8bavuYmUxkNKIgJ3-9xYldGsCg5mpKPDW4LZw%7ExdVp1pBtCdm473fTXPn7X1aNCSLBhRMM%7EWhmhmeh-tNHcVFDpCJX182yxh89vA__&Key-Pair-Id=K24J24Z295AEI9"

python feature_extract.py --input_path [your_image_root] --output_path [your_npy_root] --backbone_ckpt_path ./aim_3b_5bimgs_attnprobe_backbone.pth --head_ckpt_path ./aim_3b_5bimgs_attnprobe_head_best_layers.pth
```

## Reconstruct Real Images
We use vae in stable diffusion to reconstruct images. You can also use other generative models.
```bash
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

# you can also use HF-mirror to clone
# git clone https://hf-mirror.com/CompVis/stable-diffusion-v1-4

python reconstruct.py --repo_id ./stable-diffusion-v1-4 --input_dir [your_real_image] --output_dir [your_reconstruct_dir]
```

## Train
We have extracted features of 10,000 real images and their reconstructed images, which are stored in `UAR/data` and can be used directly for training. 

The origin image can be downloader from [BaiduNetdisk](https://pan.baidu.com/s/1secHnpVj0_a82vP17MCClQ?pwd=nw1h) or [Google Drive](https://drive.google.com/file/d/146GQNq3zrLIApzDFvEXx9xRc0iFC2mGs/view?usp=drive_link)
```bash
# Train
python train.py --dataroot ./data --savedir [your_save_path]
```

If you want to train on your own dataset, you can reconstruct real images using `reconstruct.py` and extract features using `feature_extract.py`
```bash
python reconstruct.py --repo_id ./stable-diffusion-v1-4 --input_dir [your_real_image_dir] --output_dir [your_reconstruct_image_dir]
python feature_extract.py --input_path [your_image_root] --output_path [your_npy_root] --backbone_ckpt_path ./aim_3b_5bimgs_attnprobe_backbone.pth --head_ckpt_path ./aim_3b_5bimgs_attnprobe_head_best_layers.pth
```

## Test
We provide test image features (.npy) in the following links:
- Baidu Netdisk: [Download test image features from BaiduNetdisk](https://pan.baidu.com/s/1jFyLZ8sFNh2pN-sX9qUQDg?pwd=23pz).
- Google Drive: [Download test image features from Google Drive](https://drive.google.com/file/d/1nsrkyfLX9dqtW7xnwTrQQhCSA3U2kgvj/view?usp=sharing).

The test images are from this [GitHub repository](https://github.com/Ekko-zn/AIGCDetectBenchmark) and can be download from [BaiduNetdisk](https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u#list/path=%2F).
The SDXL and Flux-generated images constructed by us can be downloaded from [BaiduNetdisk](https://pan.baidu.com/s/1SOvaJULLTvnIQG5EVb67Aw?pwd=htr6) or [Google Drive](https://drive.google.com/file/d/1oIw0dWOWA8xrg9VlzYc_a8nnO4frsCuw/view?usp=sharing)

```bash
# Test
python eval.py --dataroot ./test_npy --checkpoint ./checkpoints/AIMClassifier/epoch_0_model.pth
```

## Test Results
### Comparison of Average Accuracy (%) Across Different Generators and Detection Methods

| **Generator**  | **CNNDet** | **FreDect** | **Fusing** | **Gram** | **LNP** | **LGrad** | **DIRE** | **UnivFD** | **PatCra** | **NPR** | **UAR** |
|----------------|------------|-------------|------------|----------|---------|-----------|----------|------------|------------|---------|---------|
| **ProGAN**     | **100.00** | 99.36       | <u>99.99</u> | <u>99.99</u> | 99.95   | 99.83     | 52.75    | 99.81      | **100.00** | 99.79   | 92.24   |
| **StyleGAN**   | 90.17      | 78.02       | 85.19      | 87.05    | 92.64   | 91.08     | 51.31    | 84.93      | 92.77      | **97.85** | <u>95.50</u> |
| **BigGAN**     | 71.17      | 81.97       | 77.38      | 67.33    | 88.43   | 85.62     | 49.70    | 95.08      | <u>95.80</u> | 84.35   | **95.67** |
| **CycleGAN**   | 87.62      | 78.77       | 87.02      | 86.07    | 79.07   | 86.94     | 49.58    | <u>98.33</u> | 70.17      | 96.10   | **98.35** |
| **StarGAN**    | 94.60      | 94.62       | 97.02      | 95.05    | **100.00** | 99.27    | 46.72    | 95.75      | <u>99.97</u> | 99.35   | 97.15   |
| **GauGAN**     | 81.42      | 80.57       | 77.96      | 69.35    | 79.17   | 78.46     | 51.23    | **99.47**  | 71.58      | 82.50   | <u>93.72</u> |
| **StyleGAN2**  | 86.91      | 66.19       | 83.27      | 87.28    | <u>93.82</u> | 85.32     | 51.72    | 74.96      | 89.55      | **98.52** | 94.65   |
| **WFIR**       | <u>91.65</u> | 50.75       | 73.45      | 86.80    | 50.03   | 55.70     | 53.30    | 86.90      | 85.80      | 51.20   | **97.10** |
| **ADM**        | 60.39      | 63.42       | 56.57      | 58.61    | 83.91   | 67.15     | **98.25** | 66.87      | 82.17      | 86.50   | <u>90.42</u> |
| **Glide**      | 58.07      | 54.13       | 57.20      | 54.50    | 83.50   | 66.11     | 92.42    | 62.46      | 83.79      | **95.47** | <u>93.57</u> |
| **Midjourney** | 51.39      | 45.87       | 52.17      | 50.02    | 69.55   | 65.35     | 89.45    | 56.13      | <u>90.12</u> | **91.51** | 77.09   |
| **SDv1.4**     | 50.57      | 38.79       | 51.03      | 51.70    | 89.33   | 63.02     | 91.24    | 63.66      | 95.38      | <u>97.07</u> | **99.32** |
| **SDv1.5**     | 50.53      | 39.21       | 51.35      | 52.16    | 88.81   | 63.67     | 91.63    | 63.49      | 95.30      | <u>96.86</u> | **99.18** |
| **SDXL**       | 57.18      | 54.03       | 59.95      | 65.15    | 57.45   | 57.85     | 51.72    | 50.38      | <u>95.88</u> | 85.08   | **98.35** |
| **Flux**       | 48.75      | 49.50       | 52.95      | 64.15    | 67.45   | 63.20     | 51.40    | 49.85      | 79.40      | <u>80.70</u> | **80.90** |
| **VQDM**       | 56.46      | 77.80       | 55.10      | 52.86    | 85.03   | 72.99     | 91.90    | 85.31      | 88.91      | <u>95.31</u> | **99.11** |
| **wukong**     | 51.03      | 40.30       | 51.70      | 50.76    | 86.39   | 59.55     | 90.90    | 70.93      | 91.07      | <u>96.38</u> | **99.36** |
| **DALL-E**     | 50.45      | 34.70       | 52.80      | 49.25    | 92.45   | 65.45     | <u>92.45</u> | 50.75      | 96.60      | **98.25** | 89.29   |
| **Average**    | 68.80      | 62.67       | 67.84      | 68.23    | 82.61   | 73.70     | 69.32    | 75.28      | 89.13      | <u>90.71</u> | **93.94** |



---

### Comparison of Average Precision (%) Across Different Generators and Detection Methods

| **Generator**  | **CNNDet** | **FreDect** | **Fusing** | **Gram** | **LNP** | **LGrad** | **DIRE** | **UnivFD** | **PatCra** | **NPR** | **UAR** |
|----------------|------------|-------------|------------|----------|---------|-----------|----------|------------|------------|---------|---------|
| **ProGAN**        | **100.00** | <u>99.99</u>  | **100.00** | **100.00** | **100.00** | **100.00** | 58.79    | **100.00** | **100.00** | <u>99.99</u> | 99.96   |
| **StyleGAN**       | <u>99.83</u>  | 88.98        | 99.48      | 99.23    | 99.27   | 98.31     | 56.68    | 97.56      | 98.96      | **99.92** | 99.75   |
| **BigGAN**         | 85.99      | 93.62        | 90.75      | 81.79    | 94.54   | 92.93     | 46.91    | 99.27      | <u>99.42</u> | 87.80    | **99.84** |
| **CycleGAN**       | 94.94      | 84.78        | 95.48      | 95.33    | 89.52   | 95.01     | 50.03    | <u>99.80</u> | 85.26      | 98.45    | **99.93** |
| **StarGAN**        | 99.04      | 99.49        | 99.82      | 99.23    | **100.00** | **100.00** | 40.64    | 99.37      | **100.00** | <u>99.94</u> | <u>99.94</u> |
| **GauGAN**         | 90.82      | 82.84        | 88.31      | 84.99    | 84.54   | 95.43     | 47.34    | **99.98**  | 81.33      | 85.49    | <u>99.88</u> |
| **StyleGAN2**      | 99.48      | 82.54        | 99.56      | 99.11    | <u>99.70</u> | 97.89     | 58.03    | 97.90      | 97.74      | **99.99** | 99.21   |
| **WFIR**           | **99.85**  | 95.89        | 93.30      | 95.21    | 42.75   | 57.99     | 59.02    | 96.73      | 95.26      | 67.44    | <u>99.67</u> |
| **ADM**            | 75.67      | 61.77        | 74.87      | 73.11    | 93.37   | 72.95     | **99.79** | 86.81      | 93.40      | 95.73    | <u>98.77</u> |
| **Glide**          | 72.28      | 52.92        | 77.48      | 66.76    | 92.76   | 80.42     | **99.54** | 83.81      | 94.04      | 98.68    | <u>99.36</u> |
| **Midjourney**     | 66.24      | 46.09        | 69.90      | 56.82    | 86.92   | 71.86     | <u>97.32</u> | 74.00      | 96.48      | **97.33** | 96.21   |
| **SDv1.4**         | 61.20      | 37.83        | 65.30      | 59.83    | 96.34   | 62.37     | 98.61    | 86.14      | <u>99.06</u> | 98.81    | **99.99** |
| **SDv1.5**         | 61.56      | 37.76        | 65.63      | 60.37    | 96.00   | 62.85     | 98.83    | 85.84      | <u>99.06</u> | 98.59    | **99.98** |
| **SDXL**           | 66.94      | 56.19        | 83.30      | 63.18    | 60.90   | 58.88     | 54.22    | 56.81      | <u>99.26</u> | 91.29    | **99.88** |
| **Flux**           | 47.06      | 49.75        | 64.24      | 60.83    | 73.82   | 66.03     | 47.61    | 49.22      | <u>89.53</u> | 85.31    | **96.77** |
| **VQDM**           | 68.83      | 85.10        | 75.44      | 61.13    | 94.91   | 77.47     | <u>98.98</u> | 96.53      | 96.26      | 98.25    | **99.97** |
| **wukong**         | 57.34      | 39.58        | 64.51      | 55.62    | 95.33   | 62.48     | 98.37    | 91.07      | 97.54      | <u>98.59</u> | **99.99** |
| **DALL-E**         | 53.51      | 38.20        | 68.13      | 49.82    | 98.26   | 82.55     | **99.71** | 63.04      | <u>99.56</u> | 99.30    | 99.02   |
**Average**     | 80.41      | 67.96        | 82.11      | 75.69    | 88.83   | 79.75     | 72.80    | 86.88      | <u>95.68</u> | 94.49    | **99.34** |

---














