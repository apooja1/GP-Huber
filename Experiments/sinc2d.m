clc
clear all 
close all 


% rng('default')
sinc = @(x1,x2) sin(sqrt(x1.^2+x2.^2))./(sqrt(x1.^2+x2.^2));

n=100;

for i=1:n
noise(i,1)=0.01*trnd(2,1,1);
end

a=-8; 
b=8;
x1=a + (b-a).* rand(n,1);
x2=a + (b-a).* rand(n,1);

y=sinc(x1,x2);

out_idx_v1=[20; 21; 22; 23; 24;25;26;27;28;29];
% out_idx_v2=[48; 49; 50; 51; 52; 53; 54; 55];%   100; 15;10;11;12;13;15];
% % 
% % adding vertical outliers
y(out_idx_v1)=0.1*[8.6;8.7;8.8;8.9;8.10;8.11;8.12;8.13;8.14;8.15];             
% x1(out_idx_v2)=[12.1;12.2;12.3;12.4;12.5;12.6;12.7;12.8];

y=y+noise;

x=[x1 x2];

nstar=200;
a=-16; 
b=16; 
n=100;
xt1=a + (b-a).* rand(nstar,1);
xt2=a + (b-a).* rand(nstar,1);

yt=sinc(xt1,xt2);
xt=[xt1 xt2];

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

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft=sqrt(Varft);

model=1;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

% Plot the network outputs 
mu=Eft; s2=std_ft;
N = 1000; %length(mu);
X=xt1;
Y=xt2;
Z=mu;
xv = linspace(min(X), max(X), N);
yv = linspace(min(Y), max(Y), N);
[Xm,Ym] = ndgrid(xv, yv);
Zm = griddata(X,Y,Z,Xm,Ym);

figure(1)
scatter3(x1,x2,y,12,[1 0 0],"filled","MarkerEdgeColor",[0 0 0]) %red
hold on
scatter3(xt1,xt2,yt,12,[0 1 0],"filled","MarkerEdgeColor",[0 0 0]) %green
% scatter3(x1(out_idx_v2),x2(out_idx_v2),y(out_idx_v2),12,[0 0 1],"filled","MarkerEdgeColor",[0 0 0])  %blue
s = surf(Xm,Ym,Zm,"FaceColor",'interp', 'EdgeColor','none');
hold off
    



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


model=2;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
% Plot the prediction and data
% plot the training data with dots and the underlying 
% mean of it as a line

% Plot the prediction and data
mu=Eft; s2=std_ft;
N = 1000 %length(mu);
X=xt1;
Y=xt2;
Z=mu;
xv = linspace(min(X), max(X), N);
yv = linspace(min(Y), max(Y), N);
[Xm,Ym] = ndgrid(xv, yv);
Zm = griddata(X,Y,Z,Xm,Ym);

figure(2)
scatter3(x1,x2,y,12,[1 0 0],"filled","MarkerEdgeColor",[0 0 0])
hold on
scatter3(xt1,xt2,yt,12,[0 1 0],"filled","MarkerEdgeColor",[0 0 0])
% scatter3(x1(out_idx_v2),x2(out_idx_v2),y(out_idx_v2),12,[0 0 1],"filled","MarkerEdgeColor",[0 0 0])
s = surf(Xm,Ym,Zm,"FaceColor",'interp', 'EdgeColor','none');
hold off
