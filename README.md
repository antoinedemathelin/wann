# WANN

Weighting Adversarial Neural Network

WANN is a supervised domain adaptation method suited for regression tasks. The algorithm is an instance-based method which learns a reweighting of source instance losses in order to correct the difference between source and target distributions.

## Requirements

Code for the numerical experiments requires the following packages:
- `tensorflow` (>= 2.0)
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `nltk`
- [`adapt`](https://github.com/adapt-python/adapt)

## Experiments

WANN algorithm is compared to several domain adaptation base-lines:
  - KMM [Huang et al.](http://papers.nips.cc/paper/3075-correcting-sample-selection-bias-by-unlabeled-data.pdf)
  - KLIEP [Sugiyama et al.](https://papers.nips.cc/paper/3248-direct-importance-estimation-with-model-selection-and-its-application-to-covariate-shift-adaptation.pdf)
  - TrAdaBoostR2 [Pardoe et al.](http://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-ICML10-pardoe.html)
  - DANN [Ganin et al.](https://arxiv.org/pdf/1505.07818.pdf)
  - ADDA [Tzeng et al.](https://arxiv.org/pdf/1702.05464.pdf)
  - MDD [Zhang et al.](https://arxiv.org/pdf/1904.05801.pdf)

The implementation of WANN can be found in the `wann\methods` folder.

The experiments are conducted on one synthetic and two benchmark datasets:
- [CityCam](https://www.citycam-cmu.com/dataset)
- [Sentiment analysis](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/)


