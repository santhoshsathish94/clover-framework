"""Runs the released learning-concepts train.py with another seed (train.py fixes TrainingConfig.seed = 42 and has no
seed flag; the paper reports means over three seeds). Nothing else changes.
Usage (from the learning-concepts folder): LC_SEED=43 python run_train_seed.py <train.py arguments>"""
import dataclasses
import os
import sys

sys.path.insert(0, os.getcwd())
import train  # noqa: E402  (the released train.py)

seed = int(os.environ["LC_SEED"])
_released = train.TrainingConfig
train.TrainingConfig = lambda: dataclasses.replace(_released(), seed=seed)
sys.argv = ["train.py"] + sys.argv[1:]
train.main()
