function [ap,firstRank,rec,prec] = getAP(confidences,overlaps,ordering)
    % Computes average precision using the PASCAL method of computation.
    [~,order] = sort(confidences(:),ordering);
    firstRank = rem(sum(~cumsum(overlaps'))+1,1+size(overlaps,2));
    tpRows = find(firstRank);
    pos = sub2ind(size(overlaps),tpRows,firstRank(tpRows));
    tp = zeros(size(overlaps));
    tp(pos) = 1;
    tp = tp(order);
    fp = ~tp;
    
    npos = size(overlaps,1);

    % Rest taken from PASCAL VOC code
    fp=cumsum(fp);
    tp=cumsum(tp);
    rec=tp/npos;
    prec=tp./(fp+tp);

    % compute average precision

    ap=0;
    for t=0:0.1:1
        p=max(prec(rec>=t));
        if isempty(p)
            p=0;
        end
        ap=ap+p/11;
    end
end

