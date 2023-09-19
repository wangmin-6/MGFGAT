# MGFGAT
# Alzheimer's Disease Diagnosis with Multi-Graph Fusion and Graph Attention Networks


See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```
**PyTorch Geometric**

To install PyTorch Geometric library, [please refer to the pyg](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Dataset 
**ADNI** raw datasets can be acquired via following link [ADNI](https://adni.loni.usc.edu/):

### How to run classification?
Process the raw data through DPABSF to obtain the Functional connectivity matrix and then generate the data set through `process_data.py`.


```
MGFGAT/run/process_data.py
```

### How to run classification?
Training and testing are integrated in file `main.py`.

```
MGFGAT/run/main.py
```
