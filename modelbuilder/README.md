# Model builder

Use a [Mobilenet v2](https://www.kaggle.com/models/google/mobilenet-v2/tensorFlow2/tf2-preview-feature-vector/4) with transfer leaning to train a new image classifiction model with the dataset [Jackestar/RLCS_circuit_IEC_diagrams](https://huggingface.co/datasets/Jackestar/RLCS_circuit_IEC_diagrams)

**how to use**

create python env if necessary

```
py -m venv .
```

install dependencies
```
pip install huggingface-hub datasets tensorflow
```

run `modeltrain`
```
python modeltrain.py
```

if everything was corretly, the script generates a file named `rlcmodel.hs` in the working directory