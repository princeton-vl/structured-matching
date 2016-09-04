classdef PhraseLocalizationEval < handle
    %PHRASELOCALIZATIONEVAL Performs the detection evaluation for phrases.
    %   This is a class with abstract methods to evaluate the Flickr30k 
    %   Entities dataset via average precision and recall at k.  The two
    %   primary methods to use are the constructor, and the function 
    %   doEvaluation.  Any child class must implement the 
    %   setTestPhraseMapping function.  A child class must also set the 
    %   labels property after calling PhraseLocalizationEval constructor 
    %   (see CoarseCategoryEval or WordEval for an example).
    %
    %  Make sure you have the external directory and the matlab functions
    %  packaged with the Flickr30k Entities dataset added to the matlab
    %  path before using this class.
    
    properties
        % recall@k thresholds to compute
        recallAt = [1,5,10];
        
        % labels of the categories the evaluation is performed over
        labels = {};
        
        % A cell array of length N, with each cell containing another cell
        % array of length M where each cell contains an array of PhraseInfo
        % objects that belong to the respective label and image. 
        % N is the number of labels and M is the number of test images.
        % See CoarseCategoryEval or WordEval for an example of how this is
        % set.
        testPhraseMapping = [];
        
        % either 'ascend' (if smaller confidence scores are better) or 
        % decend (if larger scores are better)
        scoreOrdering = 'descend';
        
        % intersection over union threshold used to determine success for a
        % phrase
        successThreshold = 0.5;
        
        % intersection over union threshold for non-maximum suppression
        nmsThreshold = 0.5;
    end
    
    properties (SetAccess = private)
        % list of testing images
        testImages = {};
        
        % same size as the test images, contains the information about each
        % image found in the phraseList.txt file
        gtPhraseMapping = {};
        
        % directory where the dataset is located (assumes the internal
        % directory structure hasn't been changed)
        gtdir = 'Flickr30kEntities';
    end
    
    methods (Abstract)
        % sets the testPhraseMapping property, automatically called when
        % the labels property is set
        setTestPhraseMapping(this);
    end

    methods
        function eval = PhraseLocalizationEval(imageList,groundTruthDir)
            %% PhraseLocalizationEval(imageList,groundTruthDir) 
            %  imageList - list of test images
            %  groundTruthDir - root directory of the Flickr30k Entities dataset
            if nargin > 0
                % phraseList.txt has the following format:
                % imageID sentenceNumber phraseNumber gtTestPhraseMatch words
                % see PhraseInfo for description of each item
                fileID = fopen('phraseList.txt','r');
                C = textscan(fileID,'%s','Delimiter','\n');
                fclose(fileID);
                C = cellfun(@(f)textscan(f,'%s'),C{1},'UniformOutput',false);
                C = vertcat(C{:});
                fns = cellfun(@(f)f{1},C,'UniformOutput',false);
                testIdx = cellfun(@(f)find(strcmp(f,fns)),imageList,'UniformOutput',false);
                notFound = cellfun(@isempty,testIdx);
                testIdx(notFound) = [];
                imageList(notFound) = [];
                eval.gtPhraseMapping = cell(length(testIdx),1);
                eval.testImages = imageList;
                for i = 1:length(testIdx)
                    eval.gtPhraseMapping{i} = PhraseInfo(C(testIdx{i}));
                end

                eval.gtdir = groundTruthDir;
            end
        end
        
        function performance = doEvaluation(this,evaluationDirectory,useNMS,useGTPhrase)
            %% performance = doEvaluation(this,evaluationDirectory,useNMS,useGTPhrase) 
            %  computes recall and average precision metrics
            %
            %  this - PhraseLocalizationEval object
            %  evaluationDirectory - directory with the prediction files, see
            %                        the README for format
            %  useNMS - logical value to indicate whether to use
            %           non-maxiumum suppression or not, default is true
            %  useGTPhrase - logical value to indicate whether to use the
            %                phrases that have been exactly matched using
            %                the Stanford Parser (false), or just evaluate
            %                performance on all ground truth phrases (true)
            %
            %
            % performance - structure with the following fields:
            %                  overallRecallAtK - #recall success/#total over the
            %                                     entire dataset at various K
            %                  overallOracle - upper bound performance for
            %                                  the above metric
            %                  oracle - upper bound performance per category
            %                  nInstances - #instances per category
            %                  ap - average precision per category
            %                  recallAtK - #category recall success/#category total
            %                  K - number of items to considered when
            %                      computing recall
            %                  labels - category labels
            %
            if nargin < 3
                useNMS = true;
            end
            if nargin < 4
                useGTPhrase = false;
            end
            
            ap = zeros(length(this.testPhraseMapping),1);
            firstRank = cell(length(this.testPhraseMapping),1);
            oracle = zeros(length(this.testPhraseMapping),1);
            
            nLabels = length(this.testPhraseMapping);
            updateThresh = [0.25,0.5,0.75];
            updateAt = round(nLabels*updateThresh);
            for i = 1:nLabels
                updateIdx = find(updateAt == i,1);
                if ~isempty(updateIdx)
                    fprintf('%2.1f percent complete\n',updateThresh(updateIdx)*100);
                end
                [predictions,confidences,oracle(i)] = this.getPredictions(evaluationDirectory,useNMS,useGTPhrase,i);
                [ap(i),firstRank{i}] = getAP(confidences,predictions,this.scoreOrdering);
            end
            
            successAtK = zeros(length(firstRank),length(this.recallAt));
            for i = 1:length(firstRank)
                ranks = firstRank{i}(firstRank{i} > 0)';
                if length(ranks) > 0
                    thresholds = repmat(this.recallAt,length(ranks),1);
                    success = repmat(ranks,1,length(this.recallAt)) <= thresholds;
                    if size(success,1) > 1
                        success = sum(success);
                    end

                    successAtK(i,:) = success;
                end
            end
            
            gtCnt = cellfun(@length,firstRank);
            recallAtK = successAtK./repmat(gtCnt,1,size(successAtK,2));
            
            nTotal = sum(gtCnt);
            performance.overallRecallAtK = sum(successAtK)./nTotal;
            performance.overallOracle = sum(oracle)/nTotal;
            performance.oracle = oracle./gtCnt;
            performance.nInstances = gtCnt;
            performance.ap = ap;
            performance.recallAtK = recallAtK;
            performance.K = this.recallAt;
            performance.labels = this.labels;
        end
        
        function [isTruePositive,score,oracle] = getEntityScore(this,item,evaluationDirectory,useNMS,useGTPhrase)
            %% [isTruePositive,score,oracle] = getEntityScore(this,item,evaluationDirectory,useNMS,useGTPhrase)
            %  obtains the performance for a particular phrase
            %
            %  this - PhraseLocalizationEval object
            %  item - PhraseInfo object for the phrase to evaluate
            %  evaluationDirectory - directory with the prediction files, see
            %                        the README for format
            %  useNMS - logical value to indicate whether to use
            %           non-maxiumum suppression or not, default is true
            %  useGTPhrase - logical value to indicate whether to use the
            %                phrases that have been exactly matched using
            %                the Stanford Parser (false), or just evaluate
            %                performance on all ground truth phrases (true)
            %
            %  isTruePositive - array of N logical values indicating if the
            %                   bounding box prediction was successful or not
            %  score - array of N confidence values indicating
            %  oracle - true if success was possible for the set of
            %           boxes in the prediction file
            %
            isTruePositive = [];
            score = [];
            fn = sprintf('%s/%s_%i_%i.txt',evaluationDirectory,item.imageID,item.sentenceNumber,item.phraseNumber);
            [confidence,b1,b2,b3,b4] = textread(fn,'%f %f %f %f %f');
            boxes = [b1,b2,b3,b4];
            annotations = getAnnotations(sprintf('%s/Annotations/%s.xml',this.gtdir,item.imageID));
            sentenceData = getSentenceData(sprintf('%s/Sentences/%s.txt',this.gtdir,item.imageID));
            annoIdx = annotations.idToLabel{strcmp(item.getPhraseID(sentenceData),annotations.id)};
            gtboxes = vertcat(annotations.labels(annoIdx).boxes);
            gtboxes = [min(gtboxes(:,1)),min(gtboxes(:,2)),...
                       max(gtboxes(:,3)),max(gtboxes(:,4))];
                
            overlaps = getIOU(gtboxes,boxes) >= this.successThreshold;
            oracle = ~isempty(find(overlaps,1));
            if useGTPhrase || item.gtTestPhraseMatch
                if useNMS
                    pick = nms_iou([boxes,confidence],this.nmsThreshold,this.scoreOrdering);
                    overlaps = overlaps(pick);
                    confidence = confidence(pick);
                end
                
                [score,order] = sort(confidence',this.scoreOrdering);
                isTruePositive = overlaps(order)';
            end
        end
        
        function [predictions,confidences,oracle] = getPredictions(this,evaluationDirectory,useNMS,useGTPhrase,labelIdx)
            %% [predictions,confidences,oracle] = getPredictions(this,evaluationDirectory,useNMS,useGTPhrase,labelIdx)
            %  gets all predictions for evaluation for a given category
            %
            %  this - PhraseLocalizationEval object
            %  evaluationDirectory - directory with evaluation files
            %  useNMS - true when using nms, false otherwise
            %  useGTPhrase - true when making predictions for all ground
            %                truth phrases, disregarding the gtTestPhraseMatch
            %                flag set in each PhraseInfo
            %  labelIdx - category index in this.testPhraseMapping to
            %             evaluate
            %
            labelMapping = this.testPhraseMapping{labelIdx};
            noLabelAtIdx = cellfun(@isempty,labelMapping);            
            labelMapping(noLabelAtIdx) = [];
            predictions = cell(length(labelMapping),1);
            confidences = cell(length(labelMapping),1);
            oracle = cell(length(labelMapping),1);
            for i = 1:length(labelMapping)
                predictions{i} = cell(length(labelMapping{i}),1);
                confidences{i} = cell(length(labelMapping{i}),1);
                oracle{i} = false(length(labelMapping{i}),1);
                for j = 1:length(labelMapping{i})
                    [predictions{i}{j},confidences{i}{j},oracle{i}(j)] ...
                        = this.getEntityScore(labelMapping{i}(j),evaluationDirectory,useNMS,useGTPhrase);
                end
            end

            predictions = vertcat(predictions{:});
            confidences = vertcat(confidences{:});
            oracle = sum(cellfun(@sum,oracle));
            maxDim = max(cellfun(@(f)size(f,2),predictions));
            predictions = cellfun(@(f)horzcat(f,false(1,maxDim-size(f,2))),predictions,'UniformOutput',false);
            if strcmp(this.scoreOrdering,'descend')
                confidences = cellfun(@(f)horzcat(f,-inf(1,maxDim-size(f,2))),confidences,'UniformOutput',false);
            else
                confidences = cellfun(@(f)horzcat(f,inf(1,maxDim-size(f,2))),confidences,'UniformOutput',false);
            end

            predictions = vertcat(predictions{:});
            confidences = vertcat(confidences{:});
        end
        
        function set.labels(this,values)
            this.labels = values;
            this.setTestPhraseMapping;
        end
        
        function set.gtdir(this,gtdir)
            if ~exist(gtdir,'dir')
                error('cannot find dataset directory %s',groundTruthDir)
            end
            annodir = [gtdir,'/Annotations'];
            if ~exist(annodir,'dir')
                error('cannot find dataset internal directory %s',annodir)
            end
            sentencedir = [gtdir,'/Sentences'];
            if ~exist(sentencedir,'dir')
                error('cannot find dataset internal directory %s',sentencedir)
            end
            this.gtdir = gtdir;
        end

        function set.scoreOrdering(this,ordering)
            if ~strcmp(ordering,'ascend') && ~strcmp(ordering,'descend')
                error('Invalid score ordering input value: %s',ordering);
            end

            this.scoreOrdering = ordering;
        end
    end % methods
end % classdef
