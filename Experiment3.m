% Experiment 3 for the paper
% model selection experiment by lasso
% compare rcv and csv method.

clear

load LogisticModel.mat

efind=(sum(sum((nbeta(2:end,:,:)~=0),3),2)~=0)';

m=size(meanx,1);
xd=size(meanx,2);

z=1:1:5;

mm=numel(z); % number of total studies generated.

% generate training set
n=150;
n=100;

NN=100; % number of simulations
lam_rcv=zeros(1,NN);
lam_csv=zeros(1,NN);
err_rcv=zeros(1,NN);
err_csv=zeros(1,NN);
N_missedPre_rcv=zeros(1,NN);
N_extraPre_rcv=zeros(1,NN);
N_missedPre_csv=zeros(1,NN);
N_extraPre_csv=zeros(1,NN);
    
for ii=1:NN
    
    trainx=zeros(n*mm,xd);
    
    trainy=zeros(n*mm,1);
    
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
        trainx(1+(i-1)*n:i*n,:)=x(:,2:end);
        trainy(1+(i-1)*n:i*n)=y;
        
    end
    
    % learn distributions
    
    % download glmnet_matlab package from https://web.stanford.edu/~hastie/glmnet_matlab/
    addpath('glmnet_matlab')
    
    tbeta=zeros(xd,3,mm);
    tbeta0=zeros(3,mm);
    tnbeta=zeros(xd+1,3,mm);
    
    
    x=trainx;
    y=trainy;
    
    % common 10-fold cross validation
    CVfit=cvglmnet(x,y,'multinomial',[],[],[]);
    lam_rcv(ii)=CVfit.lambda_min;
    tind=find(CVfit.glmnet_fit.lambda==CVfit.lambda_min);
    beta_rcv=zeros(3,xd);
    tem=CVfit.glmnet_fit.beta;
    for i=1:3
        beta_rcv(i,:)=tem{i}(:,tind)';
    end
    
    efind_rcv=(sum((beta_rcv~=0),1)~=0);
    
    % CSV lambda estimation
    lams=CVfit.lambda;
    
    nlam=numel(lams);
    pprobs=zeros(nlam,1); % -log probability
    
    opts=struct('lambda',lams);
    
    for i=1:mm
        ind=ones(n*mm,1);
        ind(1+(i-1)*n:i*n,1)=zeros(n,1);
        
        x=trainx(logical(ind),:);
        y=trainy(logical(ind),:);
        fit=glmnet(x,y,'multinomial',opts);
        %fit=glmnet(x,y,'multinomial');
        
        %lams=fit.lambda;
        tind=ones(n*mm,1)-ind;
        x=trainx(logical(tind),:);
        y=trainy(logical(tind),:);
        probs=glmnetPredict(fit, x, lams , 'response');
        
        for j=1:nlam
            for k=1:n
                pprobs(j)=pprobs(j)-log(probs(k,y(k),j));
            end
        end
    end
    
    [minp lamind]=min(pprobs);% take min because -log probability
    x=trainx;
    y=trainy;
    fit=glmnet(x,y,'multinomial',opts);
    lam_csv(ii)=lams(lamind);
    beta_csv=zeros(3,xd);
    tem=fit.beta;
    for i=1:3
        beta_csv(i,:)=tem{i}(:,lamind)';
    end
    
    efind_csv=(sum((beta_csv~=0),1)~=0);
    
    err_rcv(ii)=sum(abs(efind-efind_rcv));
    
    err_csv(ii)=sum(abs(efind-efind_csv));
    
    % number of missed predictors and number of extra selected predctors
    % with zero effect
    
    N_missedPre_rcv(ii)=sum((efind-efind_rcv)==1);
    N_extraPre_rcv(ii)=sum((efind-efind_rcv)==-1);
    
    N_missedPre_csv(ii)=sum((efind-efind_csv)==1);
    N_extraPre_csv(ii)=sum((efind-efind_csv)==-1);
    
    
end

% save ('model_selection_result.mat', 'lam_rcv','lam_csv','err_rcv','err_csv')


%nbins=0.5:1:30.5;
figure;
Xrcv=[N_missedPre_rcv' N_extraPre_rcv'];
hist3(Xrcv,'Ctrs',{0:1:30 0:1:5})
xlabel('$M_{rcv}$','Interpreter','latex')
ylabel('$N_{rcv}$','Interpreter','latex')
zlabel('Frequency','Interpreter','latex')
title('bivariate histogram for the RCV method','Interpreter','latex')
figure;
Xcsv=[N_missedPre_csv' N_extraPre_csv'];
hist3(Xcsv,'Ctrs',{0:1:30 0:1:5})
xlabel('$M_{csv}$','Interpreter','latex')
ylabel('$N_{csv}$','Interpreter','latex')
zlabel('Frequency','Interpreter','latex')
title('bivariate histogram for the CSV method','Interpreter','latex')

% nbins=0.5:1:30.5;
% figure;histogram(err_rcv,nbins)
% xlabel('$N_{rcv}$','Interpreter','latex')
% title('histogram of $N_{rcv}$','Interpreter','latex')
% figure;histogram(err_csv,nbins)
% xlabel('$N_{csv}$','Interpreter','latex')
% title('histogram of $N_{csv}$','Interpreter','latex')
nbins=0:0.002:0.1;
figure;histogram(lam_rcv,nbins)
xlabel('$\lambda_{rcv}$','Interpreter','latex')
ylabel('Frequency','Interpreter','latex')
title('histogram of $\lambda_{rcv}$','Interpreter','latex')
figure;histogram(lam_csv,nbins)
xlabel('$\lambda_{csv}$','Interpreter','latex')
ylabel('Frequency','Interpreter','latex')
title('histogram of $\lambda_{csv}$','Interpreter','latex')
