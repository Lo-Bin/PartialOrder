% Experiment 1

clear

load LogisticModel.mat


m=size(meanx,1);
xd=size(meanx,2);
mm=10; % number of total studies generated.

% Generate study variables i.i.d. from uniform distribution
% over the 5 stduies. 

% z=ceil(rand(1,mm)*m);
z=[1 3 5 3 2 4 3 1 1 2]; % this one realization is what used in the paper

% generate n i.i.d pairs (x, y) for each z
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


jDis=zeros(n*mm,3,mm);
x=data(:,1:xd+1);
y=data(:,xd+2);

for i=1:mm   
    tz=z(i);
    mu=meanx(tz,:);
    covar=covx(:,:,tz);
    temp=exp(x*nbeta(:,:,tz));
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
hold on
plot(2:1:mm,csv(2:end));
legend('RCV error rate','CSV error rate')
xlabel('number of studies')
ylabel('classification error rate')


    
    











