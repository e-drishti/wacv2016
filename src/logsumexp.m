%
% Author: Konstantinos Bousmalis 2013 - www.doc.ic.ac.uk/~kb709
function res = logsumexp(X, dimensionToOperate)
if nargin < 2
    B = max(X);
else
    B = max(X, [], dimensionToOperate);
end
repeatedB = repmat(B, size(X)-size(B)+1);
if nargin < 2
    res = log(sum(exp(X-repeatedB)))+B;
else
    res = log(sum(exp(X-repeatedB), dimensionToOperate))+B;
end
    function r = logsumexp1D(X)
        B = max(X);
        r = log(sum(exp(X-B)))+B;
    end
end
    