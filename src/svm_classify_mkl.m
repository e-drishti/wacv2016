% Example of how to use the mklsvm for  classification
%
%

function predicted_categories = svm_classify_mkl(train_image_feats, train_labels)

  addpath('../toollp');


  nbiter=1;
  ratio=0.3;
  %C = [100 100 100];
  verbose=1;

  options.algo='svmclass'; % Choice of algorithm in mklsvm can be either
                           % 'svmclass' or 'svmreg'
  %------------------------------------------------------
  % choosing the stopping criterion
  %------------------------------------------------------
  options.stopvariation=0; % use variation of weights for stopping criterion 
  options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
  options.stopdualitygap=1; % set to 1 for using duality gap for stopping criterion

  %------------------------------------------------------
  % choosing the stopping criterion value
  %------------------------------------------------------
  options.seuildiffsigma=1e-4;        % stopping criterion for weight variation 
  options.seuildiffconstraint=0.1;    % stopping criterion for KKT
  options.seuildualitygap=0.0001;       % stopping criterion for duality gap

  %------------------------------------------------------
  % Setting some numerical parameters 
  %------------------------------------------------------
  options.goldensearch_deltmax=1e-1; % initial precision of golden section search
  options.numericalprecision=1e-8;   % numerical precision weights below this value
                                     % are set to zero 
  options.lambdareg = 1e-8;          % ridge added to kernel matrix 

  %------------------------------------------------------
  % some algorithms paramaters
  %------------------------------------------------------
  options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                     % variable in the reduced gradient method 
  options.nbitermax=500;             % maximal number of iteration  
  options.seuil=0;                   % forcing to zero weights lower than this 
  options.seuilitermax=10;           % value, for iterations lower than this one 

  options.miniter=0;                 % minimal number of iterations 
  options.verbosesvm=0;              % verbosity of inner svm algorithm 
  options.efficientkernel=1;         % use efficient storage of kernels 


  %------------------------------------------------------------------------
  %                   Building the kernels parameters
  %------------------------------------------------------------------------
  kernelt={'gaussian' 'gaussian' 'poly' 'poly' };
  kerneloptionvect={[0.1 0.5 1 5 10 20 50 100] [0.1 0.5 1 5 10 20 50 100] [1 2 3 4] [1 2 3 4]};
  variablevec={'all' 'single' 'all' 'single'};

  %kernelt={'poly'};
  %kerneloptionvect={[1]};
  %variablevec={'single'};
	%
	ypredAcc = []
	classcode=[1 2 3];
	x = train_image_feats;
	y = train_labels;
	[nbdata,dim]=size(x);
	nbtrain=floor(nbdata*ratio);
	rand('state',1);
	%[indx x_parts y_parts] = randDivide(x,y,3);
	[xapp_orig,yapp_orig,xtest,ytest,indice]=CreateDataAppTest(x, y, nbtrain,classcode);
	indices = indice.app;
	%xapp_orig = x_parts{1};
	%yapp_orig = y_parts{1};
	%indices = indx{1};
	%xtest = [x_parts{2};x_parts{3}];
	%ytest = [y_parts{2};y_parts{3}];
	save('index.mat','indices');
	save('ytest.mat','ytest');
	for currentClass=1:3
			%load([data ]);
			xapp = xapp_orig;
			yapp = yapp_orig;
			for itr=1:size(yapp)
				if yapp(itr)~=currentClass
					yapp(itr) = -1;
				else
					yapp(itr) = 1;
				end
			end
			weights = csvread('/home/aditya/honours/demo/weights.csv');
			weights = weights/25;
			C = ones(nbdata,1);
			C = C*100;
			
			for i=1:nbdata
				%C(i) = C(i) + exp(0*weights(i,currentClass));
				if y(i)~=currentClass
					other_sum =0;
					for k=1:3
						if k==currentClass
							continue
						end
						other_sum = other_sum + weights(i,k);
					end
					C(i) = C(i) *exp(100*other_sum);
				else
					C(i) = C(i) *exp(100*weights(i,currentClass));
				end
			end
			C = C(:,1);
			C = C/ norm(C);


			for i=1: nbiter
					i
					[xapp,xtest]=normalizemeanstd(xapp,xtest);
					C = C(indices); 
					[kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
					[Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
					K=mklkernel(xapp,InfoKernel,Weight,options);


					
					%------------------------------------------------------------------
					% 
					%  K is a 3-D matrix, where K(:,:,i)= i-th Gram matrix 
					%
					%------------------------------------------------------------------
					% or K can be a structure with uses a more efficient way of storing
					% the gram matrices
					%
					% K = build_efficientK(K);
					
					tic
					[beta,w,b,posw,story(i),obj(i)] = mklsvm(K,yapp,C,options,verbose);
					timelasso(i)=toc

					Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(posw,:),beta);
					ypred=Kt*w+b;
					%save('ypred.mat','ypred')
					%bc(i)=mean(sign(ypred)==ytest)
					ypredAcc = [ypredAcc,ypred];

	end
	save('ypredAcc.mat','ypredAcc');
%  classcode=[1 -1];
%  %load([data ]);
%  x = train_image_feats;
%  y = train_labels;
%	for itr=1:size(y)
%		if y(itr)~=1
%			y(itr) = -1;
%		end
%	end
%	[nbdata,dim]=size(x);
%	weights = csvread('/home/aditya/honours/demo/weights.csv');
%	weights = weights/25;
%	C = ones(nbdata,1);
%	C = C*100;
%  
%	for i=1:nbdata
%		C(i) = C(i) + exp(weights(i));
%	end
%	C = C(:,1);
%
%	nbtrain=floor(nbdata*ratio);
%  rand('state',0);
%
%  for i=1: nbiter
%      i
%      [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(x, y, nbtrain,classcode);
%      [xapp,xtest]=normalizemeanstd(xapp,xtest);
%      C = C(indice.app); 
%			[kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
%      [Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
%      K=mklkernel(xapp,InfoKernel,Weight,options);
%
%
%      
%      %------------------------------------------------------------------
%      % 
%      %  K is a 3-D matrix, where K(:,:,i)= i-th Gram matrix 
%      %
%      %------------------------------------------------------------------
%      % or K can be a structure with uses a more efficient way of storing
%      % the gram matrices
%      %
%      % K = build_efficientK(K);
%      
%      tic
%      [beta,w,b,posw,story(i),obj(i)] = mklsvm(K,yapp,C,options,verbose);
%      timelasso(i)=toc
%
%      Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(posw,:),beta);
%      ypred=Kt*w+b;
%			save('ypred.mat','ypred')
%      bc(i)=mean(sign(ypred)==ytest)

 % end;%
predicted_categories = []

end
