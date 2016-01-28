projfunction [ predicted_categories ] = hcrf_classify( train_image_feats,train_labels, test_image_feats,test_labels )
% Builds a Hidden Conditional Random Field model for edrishti
% train_image_feats: extracted features from training data
% train_labels: labels of the training data
% test_image_feats: extracted features of testing data

%load data.mat;
seed = 200;
nStates = 30;
lambdaVal = 0;

s = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(s);

%nLabels = max(train_labels);
nLabels = 30;
nXFeats = 1;
edgeStruct = UGM_HCRF_makeEdgeStruct([],nStates,1,2);
nInstances = size(train_image_feats,1);
Xnode = cell(1,nInstances);
%y=zeros(1,size(train_image_feats,1));
y_cell=[cellfun(@str2num,train_labels(1:end),'un',0).'];
y = cell2mat(y_cell);
y = y';
for instance = 1:nInstances
    Xnode{instance} = train_image_feats(instance,:)';
%    y(instance)     = train_labels(instance);
end
nodeMap = zeros(nStates, nXFeats,'int32');
featNo = 1;
for s = 1:nStates
    for f = 1:nXFeats
        nodeMap(s, f) = featNo;
        featNo = featNo + 1;
    end    
end
edgeMap = zeros(nStates, nStates, nLabels, 'int32');
for s1 = 1:nStates
    for s2 = 1:nStates
        for l = 1:nLabels
            edgeMap(s1, s2, l) = featNo;
            featNo = featNo + 1;
        end
    end
end        
labelMap = zeros(nStates, nLabels, 'int32');
for s = 1:nStates
    for l = 1:nLabels
        labelMap(s, l) = featNo;
        featNo = featNo + 1;
    end
end
nParams = featNo-1;


Xedge = ones(nStates, nStates, nLabels);

w = rand(nParams, 1);
edgeStruct.nStates = nStates;
edgeStruct.nLabels = nLabels;
edgeStruct.useMex  = 1;


% Set up regularization parameters
lambda = lambdaVal*ones(size(w));
reglaFunObj = @(w)penalizedL2(w,@UGM_HCRF_NLL,lambda,Xnode,Xedge,y,nodeMap, edgeMap, labelMap, edgeStruct,@UGM_Infer_logChain);

% LBFGS to find the weights
display('Training...');
options.LS=0;
options.TolFun=1e-2;
options.TolX=1e-2;
options.Method='lbfgs';
options.Display='on';
options.MaxIter=400;
options.DerivativeCheck='off';
if ~exist('w.mat','file')
  w = minFunc(reglaFunObj,w, options);
  save('w.mat','w');
else
  load('w.mat');
end

display('Testing on 300 test sequences...');
test=test_image_feats;
test_cell =[cellfun(@str2num,test_labels(1:end),'un',0).'];
testData.labels = cell2mat(test_cell);
%testData.labels = testData.labels';
if ~exist('NLL.mat','file')
  for i = 1:size(test,1)    
      Xnode={test(i,:)'};
      for Y=1:nLabels
          NLL(i,Y) = UGM_HCRF_NLL(w,Xnode,Xedge,Y,nodeMap, edgeMap, labelMap, edgeStruct,@UGM_Infer_logChain);
      end
  end
  save('NLL.mat','NLL');
else
  load('NLL.mat');
end

[a, predictedLabels]=min(NLL,[],2);
size(predictedLabels)
size(testData.labels)
acc = numel(find(int32(predictedLabels)-int32(testData.labels)'==0))/numel(int32(testData.labels));

display(sprintf('The accuracy is %f%%', acc*100));
predicted_categories = predictedLabels;
end

