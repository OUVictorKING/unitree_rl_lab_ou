# Recreate Conda Environment

This folder contains an exported Conda environment file for `env_isaaclab_51`.

Recommended:

```bash
conda env create -f environment.yml
```

Use a different environment name:

```bash
conda env create -n new_env_isaaclab_51 -f environment.yml
```

The exported Python version is `3.11.15`.

For same-platform Conda package restoration only, `conda-explicit.txt` is also included. Use `environment.yml` for normal full environment recreation because it also includes pip packages.
