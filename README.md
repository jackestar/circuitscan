# Circuit Scan
*status:* Alfa (not functional)

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

**how to use**

create python env if necessary

```
py -m venv .
```

install dependencies
```
pip install tensorflow
```

```
python imagetest.py [classif_model_path] [ssd_model_path] [classname ...] [imagepath]
```