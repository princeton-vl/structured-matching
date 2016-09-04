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

To evaluate model of bipartite matching:

`cd src/lua`

`th test_matching.lua`

The predicted bounding boxes and corresponding scores are stored in `workspace/matching/`. Open matlab and run evaluation [codes]() from Bryan A. Plummer et al:

`p=runEval_arg('../../workspace/matching')`

To specify test file, add parameter `-file`. To evaluate model of structured matching:

`cd ../lua`

`th test_pc.lua`

The predicted bounding boxes and corresponding scores are stored in `workspace/matching_pc`

`p=runEval_arg('../../workspace/matching_pc')`

In all experiments, we reported Recall@1 in our ECCV paper.
