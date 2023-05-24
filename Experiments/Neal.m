% This script is used to obtain the results of the GP-Huber model on Neal dataset.

%% Neal data
% Model:
% f = 0.3 + 0.4x + 0.5sin(2.7x) + 1.1/(1+x^2).
close all 
clc 
clear all
rng(5)
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
out_idx_v=[7; 8; 9; 10; 11;15;61;70];
out_idx_b=[21,22,23];
out_idx_g=[50;51;52;53;54;55];

% adding bad leverage points
x(out_idx_b)=[4.3:0.1:4.5]';
y(out_idx_b)=[8.4763, 9.1938, 0.2833];

% adding vertical outliers
y(out_idx_v)=10*ones(length(out_idx_v),1);

% adding good leverage points 
x(out_idx_g)=[2.8;2.82;2.84;2.86;2.88;2.9;]';
y(out_idx_g)=ymodel(x(out_idx_g));


[n, nin] = size(x); 
%adding noise
for i=1:n
%  noise(i,1)=1*normrnd(0.01,0.08,1,1);    % Uncomment for normal noise
noise(i,1)=0.1*trnd(10,1,1);               % Uncomment for Student's-t noise with nu=10
% noise(i,1)=0.1*laprnd(1,1,0,1);          %Uncomment for Laplacian noise
% noise(i,1)=0.1*trnd(1,1,1);              %Uncomment for Student-t noise with nu=1 (Cauchy)
end

y=y+noise;

% Laying priors
pl = prior_t();
pm = prior_sqrtunif();
pn = prior_logunif();





%% Model 4 in the paper
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

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 2, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
lik = lik_huber('sigma2', 0.6^2, 'sigma2_prior', pn,'weights',weights);

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
xlim([-2.7 4.8])
  set(gcf,"Color",'w');
 set(gca,'FontSize',15,'FontWeight','bold')
saveas(gcf,'Neal_HuberMCMC','epsc')
axis on;
S2 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

%% Model 5 in the paper
disp(['Huber     noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters        '])


% Create the likelihood structure
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 4, ...
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
xlim([-2.7 4.8])
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









