% This algorithm is to learn a joint distribution of (X,Y) for each of the
% five studies.

clear

% load data
load MAdata.mat

% quantile normalization
MAall=quantilenorm(MAall);

m=5; % number of studies
ns=numel(classall); % number of patients
ng=numel(GS); % number of genes.

% gene selection based on H-test (Kruskal-Wallis test)
Pv=zeros(1,ng);

for i=1:ng
    Pv(i)=kruskalwallis(MAall(i,:),classall,'off');
end

% select Ng genes & ind is the corresponding indices

R=corrcoef(MAall');

[sPv tPvind]=min(Pv);

Thr=0.25;
cind=ones(1,ng);
ind=zeros(1,ng);
i=1;
while Pv(tPvind)<=0.05
    i=i+1;
    ind(tPvind)=1;
    cind=(cind & (abs(R(tPvind,:))<Thr));
    tPv=Pv(cind);
    [tpv tind]=min(tPv);
    tem=find(cind==1);
    tPvind=tem(tind);
end



%ind=(Pv<=sPv(Ng)); 
ma=MAall(logical(ind),:)';
Ng=size(ma,2);

% for each study, we use logistic lasso to learn its conditional
% distributions of classes given ma.

addpath('glmnet_matlab')
classall=classall';
sns=cumsum([0,StuSize]);
beta=zeros(Ng,3,m);
beta0=zeros(3,m);
for i=1:m
    
    x=ma(sns(i)+1:sns(i+1),:);
    y=classall(sns(i)+1:sns(i+1));
    fit=glmnet(x,y,'multinomial');
    beta1=fit.beta{1,1};
    beta2=fit.beta{1,2};
    beta3=fit.beta{1,3};
    ind1=(beta1~=0);
    ind2=(beta2~=0);
    ind3=(beta3~=0);
    ind=(ind1 | ind2 |ind3);
    bb=sum(ind);
    lam_ind=min(find(bb>=10));
    beta(:,1,i)=beta1(:,lam_ind);
    beta(:,2,i)=beta2(:,lam_ind);
    beta(:,3,i)=beta3(:,lam_ind);
    % intercept
    beta0(:,i)=fit.a0(:,lam_ind);
end

% learn distribution of x 
% assume normal distribution.

ind=(beta~=0);
beta_ind=(sum(sum(ind,3),2)~=0);
%max(sum((sum(ind,2)>0),3))

%beta=beta(xind,:,:);

meanx=zeros(m,Ng);
covx=zeros(Ng,Ng,m);

for i=1:m
    
    x=ma(sns(i)+1:sns(i+1),:);
    meanx(i,:)=mean(x);
    covx(:,:,i)=cov(x);
end

% estimate p(z,1:3) for z=1,2,3,4,5.
pz=zeros(m,3);
tN=10000;
nbeta=zeros(size(beta,1)+1,3,m);

for i=1:m
    mu=meanx(i,:);
    covar=covx(:,:,i);
    x=[ones(tN,1) mvnrnd(repmat(mu,tN,1),covar)];
    nbeta(:,:,i)=[beta0(:,i)' ; beta(:,:,i)];
    temp=exp(x*nbeta(:,:,i));
    temp=temp./repmat(sum(temp,2),1,3);
    pz(i,:)=mean(temp);
end

save ('LogisticModel.mat','beta','meanx','covx','beta0','pz','nbeta','beta_ind')

 
