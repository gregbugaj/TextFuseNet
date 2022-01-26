# Environment setup

```
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```


Test 

```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

```