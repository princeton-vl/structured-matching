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

`cd ../src/lua`

`th test_matching.lua`

The predicted bounding boxes and corresponding scores are stored in `workspace/matching/`. Open matlab and run evaluation [codes](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/Flickr30kPhraseLocalizationEval.tar.gz) from Bryan A. Plummer et al:

`p=runEval_arg('../../workspace/matching')`

To specify test file, add arguments `-file`. To evaluate model of structured matching:

`cd ../lua`

`th test_pc.lua`

The predicted bounding boxes and corresponding scores are stored in `workspace/matching_pc`

`p=runEval_arg('../../workspace/matching_pc')`

In all experiments, we reported Recall@1 in our ECCV paper.

To train your own models, you need generate features for phrases and bounding boxes. You can download our features (63G) by:

`cd workspace/`

`./fetch_train_feat.sh`

Then go into `src/lua` and run:

`th train_matching.lua`

to train a model for bipartite matching, or run:

`th train_matching_pc.lua`

to train a model for structured matching. Output models are stored in `workspace/model`. Please refer to commends to adjust learning parameters.

To generate your own features, you need 

- Download [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) dataset.
- Download [Flickr30kEntities](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/) dataset.
- Generate bounding boxes.
- Extract features for bbox.
- Extract features for phrases.

The folowing tools are used to features in our experiments:

- [Edgebox](https://github.com/pdollar/edges) (generate bounding boxes)
- [Fast-rcnn](https://github.com/rbgirshick/fast-rcnn) (image features)
- [Fisher vector](https://owncloud.cs.tau.ac.il/index.php/s/vb7ys8Xe8J8s8vo) (textual features)

Please let me know if you have any issues about our codes.

## Acknowledgements ##

Thanks to Geoff Leyland for providing an excellent implementation of simplex algorithms in lua.
