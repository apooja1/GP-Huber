% This script is used to compare the performance of the GP-Huber model with comparables on Neal dataset.
%% Neal data

% b=0.5 only for Laplace

% Model:
% f = 0.3 + 0.4x + 0.5sin(2.7x) + 1.1/(1+x^2).
close all 
clc 
clear all
% rng(5)
% s = rng
S = which('demo_regression_robust');
L = strrep(S,'demo_regression_robust.m','demodata/odata.txt');
x = load(L);
y = x(1:100,2);
x = x(1:100,1);

% Test data
xt = [-2.7:0.01:4.8]';
ymodel=@(x) 0.3+0.4*x+0.5*sin(2.7*x)+1.1./(1+x.^2);
yt = 0.3+0.4*xt+0.5*sin(2.7*xt)+1.1./(1+xt.^2);
X=x;
% out_idx_v=[7; 8; 9; 10; 11];
out_idx_v=[7; 8; 9; 10; 11;15;61;70];
out_idx_b=[21,22,23];
out_idx_g=[50;51;52;53;54;55];

% adding bad leverage points
% x(out_idx_b)=[4.3:0.1:4.5]';
% y(out_idx_b)=[8.4763, 9.1938, 0.2833];
% 
% % adding vertical outliers
% y(out_idx_v)=10*ones(length(out_idx_v),1);
% 
% % adding good leverage points 
% x(out_idx_g)=[2.8;2.82;2.84;2.86;2.88;2.9;]';
% y(out_idx_g)=ymodel(x(out_idx_g));
[n, nin] = size(x); 

%adding noise
for i=1:n
% noise(i,1)=1*normrnd(0.01,0.08,1,1);   % Uncomment for normal noise
noise(i,1)=trnd(10,1,1);               % Uncomment for Student's-t noise with nu=10
% noise(i,1)=0.1*laprnd(1,1,0,1);        %Uncomment for Laplacian noise
% noise(i,1)=0.1*trnd(1,1,1);              %Uncomment for Student-t noise with nu=1 (Cauchy)
end

y=y+noise;

% Laying priors
pl = prior_t();
pm = prior_sqrtunif();
pn = prior_logunif();

%% Model 1
% ========================================
% MCMC approach with scale mixture noise model (~=Student-t)
% Here we sample all the variables 
%     (lengthScale, magnSigma, sigma(noise-t) and nu)
% ========================================
disp(['Scale mixture Gaussian (~=Student-t) noise model                ';...
      'using MCMC integration over the latent values and parameters    '])

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);  %covariance structure
% Here, set own Sigma2 for every data point
lik = lik_gaussiansmt('ndata', n, 'sigma2', repmat(0.2^2,n,1), ...
                      'nu_prior', prior_logunif());    %Student-t scale mixture model
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9);   % positive jitter perhaps the nugget element

% Sample
[r,g,opt]=gp_mc(gp, x, y, 'nsamples', 300, 'display', 20); 

% thin the record
rr = thin(r,100,2);

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft=sqrt(Varft);

model=1;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

% Plot the network outputs as '.', and underlying mean with '--'
  figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-2.7 4.8])
 set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_SCtMCMC.png')
axis on;
S2 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))



%% Model 2
disp(['Student-t noise model with nu= 4 and using MCMC integration';...
      'over the latent values and parameters                      '])

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
pn = prior_logunif();
lik = lik_t('nu', 4, 'nu_prior', [], 'sigma2', 0.2^2, 'sigma2_prior', pn);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4, ...
             'latent_method', 'MCMC');
f=gp_pred(gp,x,y,x);
gp=gp_set(gp, 'latent_opt', struct('f', f));

% Sample 
[rgp,g,opt]=gp_mc(gp, x, y, 'nsamples', 400, 'display', 20);
rr = thin(rgp,100,2);

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft = sqrt(Varft);

% Plot the network outputs as '.', and underlying mean with '--'
figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
%   legend boxoff
xlim([-2.7 4.8])
set(gcf,"Color",'w');
set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_SCt4MCMC.png')
axis on;

S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

model=2;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
%% Model 3
disp(['Student-t noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters        '])

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
pn = prior_logunif();
lik = lik_t('nu', 4, 'nu_prior', pn, ...
            'sigma2', 0.2^2, 'sigma2_prior', pn);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-6, ...
            'latent_method', 'Laplace');

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt);

% Predictions to test points
[Eft, Varft] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);

% Plot the prediction and data
figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-2.7 4.8])
set(gcf,"Color",'w');
set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_tLA.png')
axis on;

S3 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)
model=3;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
 
        

%% Model 4
% RPR
% ========================================
% MCMC approach with Huber observation noise model
% ========================================
weights=zeros();
 b=0.5;
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

gpcf = gpcf_sexp('lengthScale', 0.2^2, 'magnSigma2', 0.2, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
lik = lik_huber('sigma2', 0.2^2, 'sigma2_prior', pn,'weights',weights);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
             'latent_method', 'MCMC');
f=gp_pred(gp,x,y,x);
gp=gp_set(gp, 'latent_opt', struct('f', f) );

% Sample 
[rgp,g,opt]=gp_mc(gp, x, y, 'nsamples', 400, 'display', 20);
rr = thin(rgp,100,2);

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft = sqrt(Varft);

model=4;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-2.7 4.8])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_HuberMCMC.png')
axis on;
S2 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

%% Model 5
disp(['Huber     noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters        '])


% Create the likelihood structure
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 4, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
% lik = lik_huber('sigma2', 0.2^2, 'sigma2_prior', pn,'weights',weights,'b',0.5,'epsilon',0.45);
lik = lik_huber('sigma2', 1^2, 'sigma2_prior', pn,'weights',weights);
% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
            'latent_method', 'Laplace');

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt,'loss','loo');
% gp=gp_optim(gp,x,y,'opt',opt);
% Predictions to test points
[Eft, Varft] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);
%% 
% Plot the prediction and data
figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-2.7 4.8])
set(gcf,"Color",'w');
set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_HuberLA.png')
axis on;

S3 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

model=5;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

%% Model 6
% Gaussian model 
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);
lik = lik_gaussian('sigma2', 0.2^2, 'sigma2_prior', pn);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9)

% --- MAP estimate ---
disp('Gaussian noise model and MAP estimate for parameters')

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt);

% Prediction
[Eft, Varft, lpyt, Eyt, Varyt] = gp_pred(gp, x, y, xt, 'yt', ones(size(xt)));
std_ft = sqrt(Varft);


model=6;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
% Plot the prediction and data
% plot the training data with dots and the underlying 
% mean of it as a line

figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-2.7 4.8])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_GP.png')
axis on;



S1 = sprintf('length-scale: %.3f, magnSigma2: %.3f  \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)


%% Model 7
% Laplace Observation model with MCMC

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
lik = lik_laplace('scale', 0.2, 'scale_prior', pn);

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

model=7;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

% Plot the network outputs as '.', and underlying mean with '--'
figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-2.7 4.8])
set(gcf,"Color",'w');
set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_LaplaceMCMC.png')
axis on;
S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

%%  Model 8
disp(['Laplace     noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters          '])


% Create the likelihood structure
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);


lik = lik_laplace('scale', 0.2, 'scale_prior', pn);
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
            'latent_method', 'EP');

opt=optimset('TolFun',1e-3,'TolX',1e-3);
gp=gp_optim(gp,x,y,'opt',opt);
[Eft, Varft] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);

% Plot the prediction and data
figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-2.7 4.8])
set(gcf,"Color",'w');
set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_LaplaceEP.png')
axis on;

S3 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

model=8;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);







