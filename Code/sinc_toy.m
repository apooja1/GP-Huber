%This script demonstrates the proposed GP-Huber on sinc toy model described
%in the introduction section.

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




pl = prior_t();
pm = prior_sqrtunif();
pn = prior_logunif();



%% Model 4
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

