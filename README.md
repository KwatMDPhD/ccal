<p align='center'>
  <img src='media/ccal_logo.png' height=180 />
</p>

<p align='center'>
  Library for hunting cancers :crab::gun:
</p>

# See big picture

CCAL itself doesn't have much code. Instead, CCAL uses other cool libraries as submodules in `ccal/`, and import their awesome functions inside `ccal/__init__.py`'.

# Get started

Two options:

1. Create conda environment for ccal (recommended)

```
git clone https://github.com/UCSD-CCAL/ccal.git
cd ccal
conda env create -f environment.yaml
```

2. Pip install CCAL and get r dependencies with conda

```
pip install ccal
conda install -c r ryp2 r-mass
```
