classdef WordEval < PhraseLocalizationEval
    %WORDEVAL Evaluation based on the phrases (after some filtering based 
    %on parts of speech) in the Flickr30k Entities dataset.
    
    methods
        function eval = WordEval(imageList,groundTruthDir)
            eval = eval@PhraseLocalizationEval(imageList,groundTruthDir);
            % setting labels calls setTestPhraseMapping which uses values
            % set in the parent class constuctor
            words = cellfun(@(f)({f.groundTruthPhrase}),eval.gtPhraseMapping,'UniformOutput',false);
            eval.labels = unique(horzcat(words{:}));
        end
        
        function setTestPhraseMapping(this)
            this.testPhraseMapping = cellfun(@(f)cell(length(this.gtPhraseMapping),1),this.labels,'UniformOutput',false);
            for i = 1:length(this.gtPhraseMapping)
                for j = 1:length(this.gtPhraseMapping{i})
                    labelIdx = strcmp(this.gtPhraseMapping{i}(j).groundTruthPhrase,this.labels);
                    this.testPhraseMapping{labelIdx}{i} = [this.testPhraseMapping{labelIdx}{i};this.gtPhraseMapping{i}(j)];
                end
            end
        end
    end % methods
end % classdef

