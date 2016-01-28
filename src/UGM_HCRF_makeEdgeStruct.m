function [edgeStruct] = UGM_HCRF_makeEdgeStruct(adj,nStates,useMex,maxIter)
% [edgeStruct] = UGM_HCRF_makeEdgeStruct(adj,nStates,useMex,maxIter)
%
% adj - nNodes by nNodes adjacency matrix (0 along diagonal)
% In this toolbox, HCRFs are implemented only with the same 
%

if nargin < 3
    useMex = 1;
end
if nargin < 4
    maxIter = 100;
end

edgeStruct.useMex  = useMex;
edgeStruct.maxIter = int32(maxIter);


