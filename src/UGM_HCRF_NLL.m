% UGM_HCRF_NLL
%
% Input:
% - Xnode is a cell of instances that contains the data
% - Xedge is a matrix with the edge features(set to 1 here)
% - Y     is an array of labels
% - nodeMap, edgeMap, labelMap are maps for our features to w
%
% The negative conditional log likelihood and the gradient
%
% HCRF paper: Quattoni et al. Hidden Conditional Random Fields, TPAMI 2007
%
% Author: Konstantinos Bousmalis 2013 - www.doc.ic.ac.uk/~kb709

function [NLL, grad] = UGM_HCRF_NLL(w,Xnode,Xedge,Y,nodeMap, edgeMap, labelMap, edgeStruct,inferFunc,varargin)
% Xnode is a cell of instances that contains the data
% Xedge is a matrix with the edge potentials
% Y     is an array of labels
% nodeMap
QCHECK = 0; % enable to run some tests while running (only for debugging)
nInstances = size(Xnode, 2);
maxState = edgeStruct.nStates;

nStates = edgeStruct.nStates;
nLabels = size(labelMap, 2);

NLL = 0;
g = cell(1,nLabels);
for iL=1:nLabels
    g{iL}=zeros(size(w));
end
grad = zeros(length(w),1);


for i = 1:nInstances
    g = cell(1,nLabels);
    for iL=1:nLabels
        g{iL}=zeros(size(w));
    end
    [nNodes, nNodeFeats]=size(Xnode{i});
    nNodeFeats = nNodeFeats + 1; % for our label
    adj=zeros(nNodes);
    for iNod = 1:nNodes-1
        adj(iNod,iNod+1) = 1;
    end
    adj = adj+adj';
    nStates = double(maxState*ones(nNodes,1));
    edgeStruct = UGM_makeEdgeStruct(adj, nStates, edgeStruct.useMex, edgeStruct.maxIter);
    
    edgeStruct.nStates=int32(edgeStruct.nStates);
    instEdgeMap = ones(maxState, maxState, edgeStruct.nEdges, 'int32');
    instXedge   = ones(1, 1, edgeStruct.nEdges, 'double');
    instNodeMap = ones(nNodes, maxState, nNodeFeats, 'int32');
    clear nodeBel; clear edgeBel; clear logNodeBel; clear logEdgeBel;
    logZ = zeros(1,nLabels);
    for iL = 1:nLabels
        for iN = 1:nNodes
            instNodeMap(iN, :, :) = [nodeMap labelMap(:, iL)];
        end
        
        for iE = 1:edgeStruct.nEdges
            instEdgeMap(:,:,iE) = edgeMap(:,:, iL);
        end
        clear instXnod;
        instXnod(1,:,:) = double([Xnode{i} ones(size(Xnode{i},1),1)]');  % include label, always active!
        
        % Make potentials
        [nodePot{iL},edgePot{iL}] = UGM_CRF_makePotentials(w, instXnod, instXedge,  instNodeMap, instEdgeMap, edgeStruct,1);
        
        % Compute marginals and logZ for the given label(the nominator)
        [nodeBel{iL},edgeBel{iL},logZ(iL)] = inferFunc(nodePot{iL}, edgePot{iL}, edgeStruct);
        
        if QCHECK
            [nodeBelCHECK{iL}, edgeBelCHECK{iL},logZCHECK(iL)] = UGM_Infer_Chain(nodePot{iL}, edgePot{iL}, edgeStruct, varargin{:});
            CHECK1 = abs(nodeBelCHECK{iL} - nodeBel{iL}) < 1e-10;
            CHECK2 = abs(edgeBelCHECK{iL} - edgeBelCHECK{iL}) < 1e-10;
            CHECK3 = abs(logZ(iL)-logZCHECK(iL)) < 1e-10;
            assert(all(CHECK1(:))); % assert that all values are approximately equal
            assert(all(CHECK2(:)));
            assert(all(CHECK3(:)));
        end
        
        nEdges   = edgeStruct.nEdges;
        edgeEnds = edgeStruct.edgeEnds;
        nEdgeFeatures = size(instXedge,2);
        
        % Update the gradients of the nLL
        % calculate the gradient for all labels E[*,*,*]
        if nargout > 1
            for n = 1:nNodes
                for s = 1:nStates(n)
                    for f = 1:nNodeFeats
                        if instNodeMap(n,s,f) > 0
                            g{iL}(instNodeMap(n,s,f)) = g{iL}(instNodeMap(n,s,f)) + nodeBel{iL}(n,s)*double(instXnod(1,f,n));
                        end
                    end
                end
            end
            for e = 1:nEdges
                n1 = edgeEnds(e,1);
                n2 = edgeEnds(e,2);
                for s1 = 1:nStates(n1)
                    for s2 = 1:nStates(n2)
                        for f = 1:nEdgeFeatures
                            if instEdgeMap(s1,s2,e,f) > 0
                                g{iL}(instEdgeMap(s1,s2,e,f)) = g{iL}(instEdgeMap(s1,s2,e,f)) + double(instXedge(1,f,e))*edgeBel{iL}(s1,s2,e);
                            end
                        end
                    end
                end
            end
        end
        
    end
    % Update NLL
    logSumZ = logsumexp(logZ);
    NLL = NLL - (logZ(Y(i))-logSumZ);
    
    for iL = 1:nLabels
        p_y(iL) = exp(logZ(iL)-logSumZ);
        grad = grad + g{iL} * p_y(iL);
    end
    grad = grad - g{Y(i)};
    clear g;
end
end