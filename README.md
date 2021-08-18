# SIDM-fitting-profiles

SIDM-fitting-profiles is a python package that allows the fitting of 
dark matter density profiles from cosmological self-interacting simulations.

# Usage

Modify run.sh and type ``bash run.sh``

# Installing

To get started using the package you need to set up a python virtual 
environment. The steps are as follows:

Clone SIDM-fitting-profiles

```git clone https://github.com/correac/SIDM-fitting-profiles.git```

```cd SIDM-fitting-profiles```

```python3 -m venv sidmfits_env```

Now activate the virtual environment.

```source sidmfits_env/bin/activate```

Update pip just in case

```pip install pip --upgrade```

```pip install -r requirements.txt```