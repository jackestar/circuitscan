# Circuit Scan
*status:* Alfa 

Optical recognition system for electronic symbols, focused on the recognition of electronic circuits for export in SPICE format

## Objectives
*optional* ~~not for now~~
* Make an image recognition model for electronic component diagrams (IEC)
* Recognize the connections between electronic components, in nodes
* *Recognize the location of the components and vectorize the connections*
* ~~*Recognize the values ​​of the components as well as any useful label (text) in it.*~~
    * Value entry for each recognized component
* Generate a format that is read by a circuit simulator (SPICE)
* Simulate


**How to Use**

create python env if necessary

```
py -m venv .
```

install dependencies
```
pip install install huggingface-hub datasets tensorflow==2.15.0 opencv-python
```

```
python imagetest.py [classif_model_path] [ssd_model_path] [classname ...] [imagepath]
```

## Issues

If you have compatibility problems with tensorflow with respect to other libraries, consider installing libraries from around Nov 14, 2023.

The code can easily run in Google Colab, because it was worked from said environment