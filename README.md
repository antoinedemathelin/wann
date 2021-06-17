# WANN

Weighting Adversarial Neural Network

An online demo of the algorithm is available at https://antoinedemathelin.github.io/demo/



WANN is a supervised domain adaptation method suited for regression tasks. The algorithm is an instance-based method which learns a reweighting of source instance losses in order to correct the difference between source and target distributions.


## Requirements

Code for the numerical experiments requires the following packages:
- `tensorflow` (>= 2.0)
- `scikit-learn`
- `numpy`
- `cvxopt`
- `nltk` (for sentiment analysis pre-processing)
- `matplotlib` (for visualization)

The file `environment.yml` can be used to reproduce the same conda environment as the one used to conduct the experiments with the following command line:

`$ conda env create -f environment.yml`

> :warning: The environment has been built on Windows, it seems that the above command line does not work on Ubuntu. If you use this operating system, please create a new environment and install the above packages using `conda install` or `pip install`.

## Experiments

WANN algorithm is compared to several instances-based domain adaptation base-lines:
  - KMM [Huang et al.](http://papers.nips.cc/paper/3075-correcting-sample-selection-bias-by-unlabeled-data.pdf)
  - KLIEP [Sugiyama et al.](https://papers.nips.cc/paper/3248-direct-importance-estimation-with-model-selection-and-its-application-to-covariate-shift-adaptation.pdf)
  - TrAdaBoostR2 [Pardoe et al.](http://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-ICML10-pardoe.html)
  - GDM [Cortes et al.](http://jmlr.org/papers/volume20/15-192/15-192.pdf)
  - DANN [Ganin et al.](https://arxiv.org/pdf/1505.07818.pdf)

The implementation of the methods can be found in the `wann\methods` folder. For GDM, code can be found at https://cims.nyu.edu/~munoz/ 

The experiments are conducted on one synthetic and three benchmark datasets:
- Superconductivity [UCI](https://archive.ics.uci.edu/ml/datasets/superconductivty+data#)
- Kin 8xy family [Delve project](http://www.cs.toronto.edu/~delve/data/datasets.html)


### Superconductivity Experiments

Running superconductivity experiments can be done in two ways:
- In the command line with: `$ python wann\uci_experiments.py`
- Within the following notebooks: `notebooks\UCI_experiments.ipynb`


### Kin Experiments

Running kin experiments can be done in two ways:
- In the command line with: `$ python wann\kin_experiments.py`
- Within the following notebooks: `notebooks\Kin_experiments.ipynb`


