clc
clear all 
close all 

rng('default')
sinc = @(x) sin(x)./x;
n=150;

for i=1:n
noise(i,1)=0.1*trnd(10,1,1);
end

a=-10; 
b=10;
x=a + (b-a) .* rand(n,1);
y=sinc(x);

% out_idx_v=[20; 21; 22; 23; 24;25];
out_idx_v=[50; 51; 52; 53; 54;55];

% adding vertical outliers
y(out_idx_v)=5*ones(length(out_idx_v),1);


y=y+noise;
xt=-b:0.05:b; xt=xt';
yt=sinc(xt);



%% 

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
% 
% figure 
% subplot(2,2,1)
% hist(rr.lik.nu,20)
% title('Mixture model, \nu')
% subplot(2,2,2)
% hist(sqrt(rr.lik.tau2).*rr.lik.alpha,20)
% title('Mixture model, \sigma')
% subplot(2,2,3) 
% hist(rr.cf{1}.lengthScale,20)
% title('Mixture model, length-scale')
% subplot(2,2,4) 
% hist(rr.cf{1}.magnSigma2,20)
% title('Mixture model, magnSigma2')

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft=sqrt(Varft);

model=1;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
%%
% Plot the network outputs as '.', and underlying mean with '--'
  figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-10 10])
 set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
% saveas(gcf,'Neal_SCtMCMC','epsc')
ax=gca;
exportgraphics(ax,'Neal_SCtMCMC.eps')
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
  set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
%   legend boxoff
xlim([-10 10])
set(gcf,"Color",'w');
set(gca,'FontSize',15,'FontWeight','bold')
ax=gca;
exportgraphics(ax,'Neal_SCt4MCMC.eps')
% saveas(gcf,'Neal_SCt4MCMC','epsc')
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
  set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-10 10])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
saveas(gcf,'Neal_tLA','epsc')
axis on;

S3 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)
model=3;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
 

% ========================================
% EP approximation Student-t likelihood
%  Here we optimize all the variables 
%   (lengthScale, magnSigma2, sigma(noise-t) and nu)
% ========================================
% disp(['Student-t noise model using EP integration over the';...
%       'latent values and MAP estimate for parameters      '])
% 
% gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1, ...
%                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);
% % Create the likelihood structure
% pnu = prior_loglogunif();
% lik = lik_t('nu', 4, 'nu_prior', pnu, 'sigma2', 0.2^2, ...
%             'sigma2_prior', pn);
% 
% % ... Finally create the GP structure
% gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4, ...
%             'latent_method', 'EP');
% 
% % --- MAP estimate ---
% 
% % Set the options for the optimization
% opt=optimset('TolFun',1e-3,'TolX',1e-3,'display','iter');
% % Optimize with the scaled conjugate gradient method
% gp=gp_optim(gp,x,y,'opt',opt);
% 
% % Predictions to test points
% [Eft, Varft] = gp_pred(gp, x, y, xt);
% std_ft = sqrt(Varft);

% Plot the prediction and data
% figure
% plot(xt,yt,'k')
% hold on
% plot(xt,Eft)
% plot(xt, Eft-2*std_ft, 'r--')
% plot(x,y,'.')
% legend('real f', 'Ef', 'Ef+-2*std(f)','y')
% plot(xt, Eft+2*std_ft, 'r--')
% title(sprintf('The predictions and the data points (Student-t noise model, (nu=%.2f,sigma=%.3f) with EP+MAP)',gp.lik.nu, sqrt(gp.lik.sigma2)));
% drawnow
% S4 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)
%                      


%% Model 4
% RPR
% ========================================
% MCMC approach with Huber observation noise model
% Here we sample all the variables 
%     (lengthScale, magnSigma, sigma(noise-t) and nu)
% ========================================
weights=zeros();
 b=1.5;
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

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 2, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
lik = lik_huber('sigma2', 0.8^2, 'sigma2_prior', pn,'weights',weights);

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
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

%%
% Plot the network outputs as '.', and underlying mean with '--'
figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-10 10])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
saveas(gcf,'Neal_HuberMCMC','epsc')
axis on;
S2 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

%% Model 5
disp(['Huber     noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters        '])


% Create the likelihood structure
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 2, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
% lik = lik_huber('sigma2', 0.2^2, 'sigma2_prior', pn,'weights',weights,'b',0.5,'epsilon',0.45);
lik = lik_huber('sigma2', 0.8^2, 'sigma2_prior', pn,'weights',weights);
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
  set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-10 10])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
saveas(gcf,'Neal_HuberLA','epsc')
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
  set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-10 10])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
saveas(gcf,'Neal_GP','epsc')
axis on;


S1 = sprintf('length-scale: %.3f, magnSigma2: %.3f  \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)



%% Gaussian mixture model with EP approximation
%  [theta,logL] = minimize(ones(nin+2,1), 'EPGPRegression',20,X,y,'mixture2Gaussian');
%  [mN,vN] = EPPrediction(X,y,xt,theta,'mixture2Gaussian');
% % [m1,m2,m0,dlnm0dnoisevar] =
% % mixture2Gaussian(y,nat1Cavity,nat2Cavity,parameter)  %for plotting
% fTest=yt;
% fold=1;
% Results(fold,1) = 1 - var(mN-fTest)/var(fTest);
% Results(fold,2) = sqrt(mean((mN-fTest).^2));
% Results(fold,3) = mean(abs(mN-fTest));
% Results(fold,4) = NLP(mN,vN, fTest);
% 
% [m0,m1,m2] = mixture2Gaussian(m,mG,vG,[pi;v1;v2]);
% like = (pi * normpdf(x,m,sqrt(v1)) +  (1-pi) * normpdf(x,m,sqrt(v2)));
% gaus = normpdf(x,mG,sqrt(vG));
% appr = normpdf(x,m1,sqrt(m2-m1^2));


%% Model 7
% Laplace Observation model  with MCMC

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

%%
% Plot the network outputs as '.', and underlying mean with '--'
figure
  mu=Eft; s2=std_ft;
  f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
  fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
  hold on; 
  plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
  plot(xt, mu,'color','r'); 
  plot(x, y, 'k.',LineWidth=1.5);
  set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-10 10])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
saveas(gcf,'Neal_LaplaceMCMC','epsc')
axis on;
S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

%%  Model 8
disp(['Laplace     noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters          '])


% Create the likelihood structure
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
% lik = lik_huber('sigma2', 0.2^2, 'sigma2_prior', pn,'weights',weights,'b',0.5,'epsilon',0.45);
lik = lik_laplace('scale', 0.2, 'scale_prior', pn);
% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
            'latent_method', 'EP');

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt);
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
  set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')

%   legend boxoff
xlim([-10 10])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
saveas(gcf,'Neal_LaplaceEP','epsc')
axis on;

S3 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

model=8;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

