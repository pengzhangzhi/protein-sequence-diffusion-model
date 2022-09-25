from setuptools import setup, find_packages

setup(
  name = 'protein-sequence-generation-with-denoising-diffusion',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models for Protein Sequence Generation - Pytorch',
  author = 'Zhangzhi Peng',
  author_email = 'pengzhangzhics@gmail.com',
  url = 'https://github.com/pengzhangzhi',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'generative models',
    "protein sequence",
  ],
  install_requires=[
    'biopython',
    'pytorch-lightning',
    # 'fair-esm',
    'accelerate',
    'einops',
    'ema-pytorch',
    'pillow',
    'torch',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
