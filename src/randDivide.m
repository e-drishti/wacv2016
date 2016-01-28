function [idxo prtA prtB]=randDivide(M,Y,K) 
	[n,m]=size(M);
	np=(n-rem(n,K))/K;
	B=M;
	[c,idx]=sort(rand(n,1));
	C=M(idx,:);
	C1 = Y(idx,:);
	i=1;
	j=1;
	prtA={};
	prtB = {};
	idxo={};
	n-mod(n,K);
	while i<n-mod(n,K)
			prtA{j}=C(i:i+np-1,:);
			prtB{j}=C1(i:i+np-1,:);
			idxo{j}=idx(i:i+np-1,1);
			i=i+np;
			j=j+1;
	end
	prtA{j}=C(i:n,:);
	prtB{j}=C1(i:n,:);
end
