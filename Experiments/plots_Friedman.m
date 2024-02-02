%Box plot for Freedman data 

% load Box_Friedman.mat
load Friedman_remse.mat
Box_Friedman=Friedman_remse;

Fried_Methods={'SCtMCMC' 'SCt4MCMC'  'tLA' 'HuberMCMC' 'HuberLA' 'GP' 'LaplaceMCMC' 'LaplaceEP'};
Fried.RMSE=squeeze(Box_Friedman(:,2,:))'; %rmse
Fried.MAE=squeeze(Box_Friedman(:,3,:))'; %mae
Fried.NLP=squeeze(Box_Friedman(:,4,:))'; %nlp

fontSize=25;
figure()
f1=figure(1);
hold on
h1 =boxplot(Fried.RMSE) 
h2 = plot(1:8, mean(Fried.RMSE),'gs','color','r')
hold off
set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',Fried_Methods)
xlabel('');
ylabel('RMSE')
set(h1(5,:),'LineWidth',2,'color','k')
set(h1(1,:),'LineWidth',2,'color','k')
set(h1(2,:),'LineWidth',2,'color','k')
set(h1(6,:),'LineWidth',2,'LineStyle',':','color','k')
set(h1(7,~(isnan(h1(7,:)))),'MarkerSize',10,'Color','k','LineWidth',1.5)
set(h2,'MarkerSize',10,'LineWidth',1.5)
% set(f1,'PaperPosition',[0.25 2.5 20 8])
% print -deps FriedBoxRMSE.eps
print -depsc -tiff -r300 -vector FriedBoxRMSE.eps

figure()
f2=figure(2);
hold on
h1 =boxplot(Fried.MAE)
h2 = plot(1:8, mean(Fried.MAE),'gs','color','r')
hold off
set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',Fried_Methods)
xlabel('');
ylabel('MAE')
set(h1(5,:),'LineWidth',2,'color','k')
set(h1(1,:),'LineWidth',2,'color','k')
set(h1(2,:),'LineWidth',2,'color','k')
set(h1(6,:),'LineWidth',2,'LineStyle',':','color','k')
set(h1(7,~(isnan(h1(7,:)))),'MarkerSize',10,'Color','k','LineWidth',1.5)
set(h2,'MarkerSize',10,'LineWidth',1.5)


figure()
f3=figure(3);
hold on
h1 =boxplot(Fried.NLP)
h2 = plot(1:8, mean(Fried.NLP),'gs','color','r')
hold off
set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',Fried_Methods)
xlabel('');
ylabel('NLP')
set(h1(5,:),'LineWidth',2,'color','k')
set(h1(1,:),'LineWidth',2,'color','k')
set(h1(2,:),'LineWidth',2,'color','k')
set(h1(6,:),'LineWidth',2,'LineStyle',':','color','k')
set(h1(7,~(isnan(h1(7,:)))),'MarkerSize',10,'Color','k','LineWidth',1.5)
set(h2,'MarkerSize',10,'LineWidth',1.5)
