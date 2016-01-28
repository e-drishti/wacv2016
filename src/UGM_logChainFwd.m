%
% Author: Konstantinos Bousmalis 2013 - www.doc.ic.ac.uk/~kb709
function [logAlpha,logZ] = UGM_logChainFwd(logNodePot, logEdgePot,nStates)
[nNodes,maxState] = size(logNodePot);

logAlpha = zeros(nNodes,maxState);
logAlpha(1,1:nStates(1)) = logNodePot(1,1:nStates(1));
for n = 2:nNodes
   logMi = repmat(logAlpha(n-1,1:nStates(n-1))',1,nStates(n)) + logEdgePot(1:nStates(n-1),1:nStates(n),n-1);
   logAlpha(n,1:nStates(n)) = logNodePot(n,1:nStates(n))+logsumexp(logMi);
end

logZ = logsumexp(logAlpha(n, 1:nStates(n)));