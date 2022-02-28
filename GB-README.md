# Environment setup

This version 1.10.1+cu111 causes error :
'The detected CUDA version (11.2) mismatches the version that was used to compile
    PyTorch (11.1). Please make sure to use the same CUDA versions.'
  

```
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Test 
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

```

Works on my version 

```
1.9.0+cu111
True
```

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


