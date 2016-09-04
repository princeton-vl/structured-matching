classdef PhraseInfo < handle
    %PHRASEINFO Stores the data for phrase localization.
    
    properties (SetAccess = private)
        % image this phrase belongs to
        imageID = [];
        
        % sentence this phrase belongs to
        sentenceNumber = [];
        
        % index of the phrase in the sentence
        phraseNumber = [];
        
        % binary indicator if the phrase was matched exactly after
        % automatically extracting them using the Standford parser
        gtTestPhraseMatch = [];
        
        % filtered phrase (used in WordEval), NO_VALID_POS indicates all
        % words were filtered out
        groundTruthPhrase = [];
    end
    
    methods
        function info = PhraseInfo(phraseInfo)
            if nargin > 0
                info(length(phraseInfo),1) = PhraseInfo;
                for i = 1:length(phraseInfo)
                    info(i).imageID = phraseInfo{i}{1};
                    info(i).sentenceNumber = str2double(phraseInfo{i}{2});
                    info(i).phraseNumber = str2double(phraseInfo{i}{3});
                    info(i).gtTestPhraseMatch = str2double(phraseInfo{i}{4});
                    words = phraseInfo{i}(5:end);
                    if length(words) > 1
                        term = cell(1,length(words)+length(words)-1);
                        term(1:2:length(term)) = words;
                        plusIdx = 2:2:length(term);
                        term(plusIdx) = repmat({' '},length(plusIdx),1);
                        words = horzcat(term{:});
                    else
                        words = words{1};
                    end
                    info(i).groundTruthPhrase = words;
                end
            end
        end
        
        function types = getPhraseTypes(this,sentenceData)
            types = sentenceData(this.sentenceNumber).phraseType{this.phraseNumber};
        end
        
        function id = getPhraseID(this,sentenceData)
            id = sentenceData(this.sentenceNumber).phraseID{this.phraseNumber};
        end
    end % methods
end % classdef
