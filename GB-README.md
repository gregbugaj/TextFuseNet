## Annotation Tool
We are going to utilize total-text dataset
https://github.com/cs-chan/Total-Text-Dataset

Tool is written in matlab so we are goint to use a Octave to run it.
```
pip3 install oct2py
```

```
from oct2py import Oct2Py
oc = Oct2Py()

script = "function y = myScript(x)\n" \
         "    y = x-5" \
         "end"

with open("myScript.m","w+") as f:
    f.write(script)

oc.myScript(7)
```


# Environment setup

```
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```


Test 

```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

```
