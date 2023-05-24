%This script is used to obtain the results of the GP-Huber model on Boston Housing dataset.
% Note: For the bar plots on Boston Housing dataset with comparables, run plots_BS

clc 
clear all 
close all

seed = 100;
rand('state',seed)
randn('state',seed)

pl = prior_t();
pm = prior_sqrtunif();
pn = prior_logunif();


for fold = 1:10
 filename =sprintf('tr%d',fold-1);
    Train = load(filename);
    x = Train(:,1:13);
    y = Train(:,14);
    [n,nin] = size(x);
    llll=5*ones(nin,1)';
    filename =sprintf('te%d',fold-1);
    Test = load(filename);
    xt = Test(:,1:13);
    yt = Test(:,14);


%% Model 4 in the paper
% ========================================
% MCMC approach with Huber observation noise model
% Here we sample all the variables 
%     (lengthScale, magnSigma, sigma(noise-t) and nu)
% ========================================
weights=zeros();
 b=5;
 X=x;
 c1=1+5/(length(X(:,1))-length(X(1,:)));
 H=[ones(length(X(:,1)),1) X];
%  H=[X y];
 [P,PS]=projectionstatistics(H);
 [m n]=size(X);
  for i=1:m
   niu=sum(H(i,:)~=0);
   cuttoff_PS(i,1)=chi2inv(0.975,niu);
   weights(i,1)=min(1,(cuttoff_PS(i,1)/PS(i)^2)); %% downweight the outliers or leverage points
  end

gpcf = gpcf_sexp('lengthScale', llll, 'magnSigma2', 0.1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
lik = lik_huber('sigma2', 2^2, 'sigma2_prior', pn,'weights',weights) %,'b',1.5,'epsilon',0.45);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
             'latent_method', 'MCMC');
f=gp_pred(gp,x,y,x);
gp=gp_set(gp, 'latent_opt', struct('f', f));

% Sample 
[rgp,g,opt]=gp_mc(gp, x, y, 'nsamples', 400, 'display', 20);
rr = thin(rgp,100,2);

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft = sqrt(Varft);

model=4;
Results(model,1,fold) = 1 - var(Eft-yt)/var(yt);  
Results(model,2,fold) = sqrt(mean((Eft-yt).^2));
Results(model,3,fold) = mean(abs(Eft-yt));
Results(model,4,fold) = NLP(Eft,Varft, yt);

%% Model 5 in the paper
disp(['Huber     noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters        '])

gpcf = gpcf_sexp('lengthScale', llll, 'magnSigma2', 0.1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
lik = lik_huber('sigma2', 2^2, 'sigma2_prior', pn,'weights',weights); %,'b',1.5,'epsilon',0.45);
% Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
            'latent_method', 'Laplace');

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt,'loss','loo');
% Predictions to test points
[Eft, Varft] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);

model=5;
Results(model,1,fold) = 1 - var(Eft-yt)/var(yt);
Results(model,2,fold) = sqrt(mean((Eft-yt).^2));
Results(model,3,fold) = mean(abs(Eft-yt));
Results(model,4,fold) = NLP(Eft,Varft, yt);



end
