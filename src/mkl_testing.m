function cm = mkl_testing(folder)
testing = 1;
	if testing == 1
		load(strcat(folder,'/','ypredAcc.mat'));
		load(strcat(folder,'/','ytest.mat'));
		load(strcat(folder,'/','index.mat'));
		predicted = ones(size(ypredAcc,1),1);
		%predicted = predicted*10;
		%ytest = ytest';
		fprintf('Size of ytest =  %d\n',size(ytest,1));
		for i=1:size(ypredAcc,1)
			maxP = log(0);
			index = 0;
			for j=1:3
				if ypredAcc(i,j) > maxP
					maxP = ypredAcc(i,j);
					index = j;
				end
			end
			predicted(i) = index;
		end
		fprintf('Unclassified points %d\n',sum(predicted==0));
		sum(ytest==predicted)
		order = [1 2 3];
		cm = confusionmat(ytest,predicted,'ORDER',order);
		cm
		fprintf('Number class 1 %d\n',sum(ytest==1));
		fprintf('Number class 2 %d\n',sum(ytest==2));
		fprintf('Number class 3 %d\n',sum(ytest==3));
		for currentClass=1:3
			ytestEach = zeros(size(ytest,1),1);
			for itr=1:size(ytest,1)
				if ytest(itr)~=currentClass
					ytestEach(itr) = -1;
				else
					ytestEach(itr) = 1;
				end
			end

			accClass = sum(sign(ypredAcc(:,currentClass))==ytestEach);
			fprintf('Accuracy for class %d is %d\n',currentClass,accClass);
		end
		return;
	end
end
