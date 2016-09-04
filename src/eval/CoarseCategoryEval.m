classdef CoarseCategoryEval < PhraseLocalizationEval
    %COARSECATEGORYEVAL Evaluation over the coarse categories in Flickr30k
    %Entities dataset.
    
    methods
        function eval = CoarseCategoryEval(imageList,groundTruthDir)
            eval = eval@PhraseLocalizationEval(imageList,groundTruthDir);
            % setting labels calls setTestPhraseMapping which uses values
            % set in the parent class constuctor
            eval.labels = {'people';'clothing';'bodyparts';'animals';'vehicles';'instruments';'scene';'other'};
        end
        
        function setTestPhraseMapping(this)
            this.testPhraseMapping = cellfun(@(f)cell(length(this.gtPhraseMapping),1),this.labels,'UniformOutput',false);
            for i = 1:length(this.gtPhraseMapping)
                sd = getSentenceData(sprintf('%s/Sentences/%s.txt',this.gtdir,this.testImages{i}));
                for j = 1:length(this.gtPhraseMapping{i})
                    phraseTypes = this.gtPhraseMapping{i}(j).getPhraseTypes(sd);
                    % sometimes a phrase can have multiple types
                    labelIdx = cellfun(@(f)find(strcmp(f,this.labels),1),phraseTypes);
                    for k = 1:length(labelIdx)
                        this.testPhraseMapping{labelIdx(k)}{i} = [this.testPhraseMapping{labelIdx(k)}{i};this.gtPhraseMapping{i}(j)];
                    end
                end
            end
        end
    end % methods
end % classdef

