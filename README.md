#Structured Matching for Phrase Localization

Created by Mingzhe Wang at University of Michigan, Ann Arbor.

This is released codes used for:

Mingzhe Wang, Mahmoud Azab, Noriyuki Kojima, Rada Mihalcea, Jia Deng
ECCV 2016. [paper](http://web.eecs.umich.edu/~jiadeng/paper/WangEtAl_ECCV2016.pdf)

To run this code, make sure the following are installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5
- Matlab

## Getting Started ##

To test with our pretrained model, run:

`cd workspace`

Download our pretrained model:

`./fetch_model.sh`

Download features extracted from testset:

`./fetch_test_feat.sh`

`cd ..`
