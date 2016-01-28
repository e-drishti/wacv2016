function  [nodeBel, edgeBel, logZ] = UGM_Infer_logChain(nodePot, edgePot, edgeStruct)
% INPUT
% nodePot(node,class)
% edgePot(class,class,edge) where e is referenced by V,E (must be the same
% between feature engine and inference engine)
%
% OUTPUT
% nodeBel(node,class) - marginal beliefs
% edgeBel(class,class,e) - pairwise beliefs
% logZ - negative of free energy
%
% NOTE: This code assumes that the edges are in order!
%		(use UGM_Infer_Tree if they are not)


[nNodes,maxState] = size(nodePot);
nEdges = size(edgePot,3);
edgeEnds = edgeStruct.edgeEnds;
nStates = edgeStruct.nStates;
maximize = 0;

logNodePot = log(nodePot);
logEdgePot = log(edgePot);
clear nodePot;
clear edgePot;

% Forward Pass
[logAlpha, logZ] = UGM_logChainFwd(logNodePot,logEdgePot,nStates);

% Backward Pass
logBeta = zeros(nNodes,maxState);
logBeta(nNodes,1:nStates(nNodes)) = 0;
logBeta(nNodes,1:nStates(nNodes)) = logBeta(nNodes,1:nStates(nNodes))-logsumexp(logBeta(nNodes,1:nStates(nNodes)));
for n = nNodes-1:-1:1
    tmp = repmat(logNodePot(n+1,1:nStates(n+1)),nStates(n),1)+logEdgePot(1:nStates(n),1:nStates(n+1),n);
    tmp2 = repmat(logBeta(n+1,1:nStates(n+1)),nStates(n),1);
    logBeta(n,1:nStates(n)) = logsumexp(tmp+tmp2,2)';
    
    % Normalize
%    logBeta(n,1:nStates(n)) = logBeta(n,1:nStates(n))-logsumexp(logBeta(n,1:nStates(n)));
end

%logAlpha = logAlpha-repmat(logsumexp(logAlpha,2),1,maxState); % norm alpha
% Compute Node Beliefs
logNodeBel = zeros(size(logNodePot));
for n = 1:nNodes
    tmp = logAlpha(n,1:nStates(n))+logBeta(n,1:nStates(n));
    logNodeBel(n,1:nStates(n)) = tmp-logsumexp(tmp);
end

% Compute Edge Beliefs
logEdgeBel = zeros(size(logEdgePot));
for n = 1:nNodes-1
    tmp = zeros(maxState);
    for i = 1:nStates(n)
        for j = 1:nStates(n+1)
            tmp(i,j) = logAlpha(n,i)+logNodePot(n+1,j)+logBeta(n+1,j)+logEdgePot(i,j,n);
        end
    end
    logEdgeBel(:,:,n) = tmp-logsumexp(tmp(:));
end

nodeBel = exp(logNodeBel);
edgeBel = exp(logEdgeBel);
end