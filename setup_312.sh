# 최신 pip
pip install pip==24.0

# PyTorch + CUDA 11.8 (3.12 지원 버전)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Lightning
pip install pytorch-lightning==2.2.4  # PL 2.x는 torch 2.x와 호환

# Huggingface transformers
pip install transformers==4.40.1

# Torchmetrics (호환 가능 최신 버전)
pip install torchmetrics==1.3.1

# 기타 유틸
pip install nltk==3.8.1
pip install jieba==0.42.1
pip install seqeval==1.2.2
pip install ark_nlp==0.0.9  # *주의: PyTorch 1.11에 최적화되어 있을 수 있음
pip install opencv-python-headless==4.9.0.80
pip install timm==0.9.12
pip install sentencepiece==0.1.99
pip install six==1.16.0
pip install textdistance==4.6.2
