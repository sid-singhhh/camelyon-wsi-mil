# CAMELYON16 metastasis detection with attention MIL

Pipeline: preprocess.py (tissue mask, patch grid, tumor labels), extract_features.py (ResNet-50 features at 20x and 10x), train_patch.py and train_mil.py (patch-level and slide-level models), heatmap.py (attention maps).

Setup (Colab, T4 GPU):
  apt-get install -y openslide-tools
  pip install openslide-python scikit-image awscli

Data (public bucket, no account needed):
  aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/images/<slide>.tif .
  Test labels: s3://camelyon-dataset/CAMELYON16/evaluation/reference.csv

Run: see run.ipynb. Slides are downloaded one at a time and deleted after feature extraction.
