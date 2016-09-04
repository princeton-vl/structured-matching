function [coarseCategoryPerformance]= runEval_arg(eva_path)
% Example code of how to evaluate localization performance using the 
% Flickr30k Entities dataset in the manner of the ICCV paper introducing
% the dataset.

% Directory where the Flickr30k Entities dataset files can be found.
% Assumes the structure hasn't been changed (i.e. has a folder called
% Annotations and Sentences within it with the annotations).
annotationDirectory = '/home/mzwang/dataset/Flickr30kEntities';

% Directory with prediction files, see the README for format
evaluationDirectory = eva_path;

addpath(annotationDirectory);
addpath('external');
load('dataSplits','testfns');

useNMS = true;

% to compare against the method in the ICCV paper, set this flag to false
useGTPhrases = false;

% if you do multiple evaluations without changing the testing filenames, it
% is recommended you save the WordEval/CoaseCategoryEval structures to save
% time and just load them when you want to do evaluations

disp('creating coarse category evaluation structure');
coarseCategory = CoarseCategoryEval(testfns,annotationDirectory);
disp('performing per category evaluation');
coarseCategoryPerformance = coarseCategory.doEvaluation(evaluationDirectory,useNMS,useGTPhrases);
