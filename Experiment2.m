% Experiment 2

clear

load LogisticModel.mat


m=size(meanx,1);
xd=size(meanx,2);

z=[1 3 5 3 2 4 3 1 1 2]; % this one realization is what used in the paper

mm=numel(z); % number of total studies generated.


% generate training set
n=150;

trainx=zeros(n,xd,mm);

trainy=zeros(n,1,mm);

for i=1:mm   
    tz=z(i);
    
    % generate x
    mu=meanx(tz,:);
    covar=covx(:,:,tz);
    x=[ones(n,1) mvnrnd(repmat(mu,n,1),covar)];
    
    % generate y
    temp=exp(x*nbeta(:,:,tz));
    py_x=temp./repmat(sum(temp,2),1,3); % probability of y given x
    temp=cumsum(py_x')';
    tr=rand(n,1);
    tr=repmat(tr,1,3);
    y=sum(temp<tr,2)+1;
    trainx(:,:,i)=x(:,2:end);
    trainy(:,:,i)=y;
    
end

% learn distributions

addpath('glmnet_matlab')

tbeta=zeros(xd,3,mm);
tbeta0=zeros(3,mm);
tnbeta=zeros(xd+1,3,mm);

tmeanx=zeros(mm,xd);
tcovx=zeros(xd,xd,mm);

for i=1:mm
    
    x=trainx(:,:,i);
    y=trainy(:,:,i);
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
    tbeta(:,1,i)=beta1(:,lam_ind);
    tbeta(:,2,i)=beta2(:,lam_ind);
    tbeta(:,3,i)=beta3(:,lam_ind);
    % intercept
    tbeta0(:,i)=fit.a0(:,lam_ind);
    tnbeta(:,:,i)=[tbeta0(:,i)' ; tbeta(:,:,i)];
    
    tmeanx(i,:)=mean(x);
    tcovx(:,:,i)=cov(x);
end


% generate n i.i.d pairs (x, y) for each z for computing classification
% error
n=10000;

data=zeros(n*mm,xd+1+1);

for i=1:mm   
    tz=z(i);
    
    % generate x
    mu=meanx(tz,:);
    covar=covx(:,:,tz);
    x=[ones(n,1) mvnrnd(repmat(mu,n,1),covar)];
    
    % generate y
    temp=exp(x*nbeta(:,:,tz));
    py_x=temp./repmat(sum(temp,2),1,3); % probability of y given x
    temp=cumsum(py_x')';
    tr=rand(n,1);
    tr=repmat(tr,1,3);
    y=sum(temp<tr,2)+1;
    data(1+(i-1)*n:i*n,:)=[x,y];   
    
end

% classification using learned distributions

jDis=zeros(n*mm,3,mm);
x=data(:,1:xd+1);
y=data(:,xd+2);

for i=1:mm   
    %tz=z(i);
    mu=tmeanx(i,:);
    covar=tcovx(:,:,i);
    temp=exp(x*tnbeta(:,:,i));
    py_x=temp./repmat(sum(temp,2),1,3); % probability of y given x
       
    px=mvnpdf(x(:,2:end), mu, covar);
    
    
    %jDis(:,:,i)=repmat(px,1,3).*py_x;
    tem=repmat(px,1,3).*py_x;
    jDis(:,:,i)=tem(1:n*mm,:);

end

%z=[1 4 1]; mm=numel(z);

rcv=zeros(1,mm);
csv=zeros(1,mm);
for k=2:mm

%% RCV error rate
tjDis=jDis(1:n*k,:,1:k);
ty=y(1:k*n);
[tMax,tind]=max(sum(tjDis,3),[],2);
rcv(k)=sum(tind~=ty)/(n*k);

%% CSV error rate

tcsv=0;
for i=1:k
    tjDis=jDis(1+(i-1)*n:i*n,:,1:k);
    ty=y(1+(i-1)*n:i*n);
    [tMax,tind]=max(sum(tjDis,3)-tjDis(:,:,i),[],2);
    tcsv=tcsv+sum(tind~=ty);
end
csv(k)=tcsv/(n*k);

end

figure; 

plot(2:1:mm,rcv(2:end));
axis([2 10 0.26 0.36])
hold on
plot(2:1:mm,csv(2:end));
legend('RCV error rate','CSV error rate')
xlabel('number of studies')
ylabel('classification error rate')


    
    











