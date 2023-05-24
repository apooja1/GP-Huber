%This script is used to obtain the bar plots for for the Boston housing
%dataset.

clc
close all 
load Results_BS_main.mat
BS_Methods={'SCtMCMC' 'SCt4MCMC'  'tLA' 'HuberMCMC' 'HuberLA' 'GP' 'LaplaceMCMC' 'LaplaceEP'};
BS.RMSE=squeeze(Results_BS_main(:,2,:))'; %rmse
BS.MAE=squeeze(Results_BS_main(:,3,:))'; %mae
BS.NLP=squeeze(Results_BS_main(:,4,:))'; %nlp

%%
f1 = figure(1)
fontSize=25;
figure()
f1=figure(1);
hold on
h1 =boxplot(BS.RMSE)
h2 = plot(1:8, mean(BS.RMSE),'gs','color','r')
hold off
set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',BS_Methods)
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
print -depsc -tiff -r300 -vector BSBoxRMSE.eps

figure()
f2=figure(2);
hold on
h1 =boxplot(BS.MAE)
h2 = plot(1:8, mean(BS.MAE),'gs','color','r')
hold off
set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',BS_Methods)
xlabel('');
ylabel('MAE')
set(h1(5,:),'LineWidth',2,'color','k')
set(h1(1,:),'LineWidth',2,'color','k')
set(h1(2,:),'LineWidth',2,'color','k')
set(h1(6,:),'LineWidth',2,'LineStyle',':','color','k')
set(h1(7,~(isnan(h1(7,:)))),'MarkerSize',10,'Color','k','LineWidth',1.5)
set(h2,'MarkerSize',10,'LineWidth',1.5)

% set(f1,'PaperPosition',[0.25 2.5 20 8])
% print -deps FriedBoxRMSE.eps
print -depsc -tiff -r300 -vector BSBoxMAE.eps

figure()
f3=figure(3);
hold on
h1 =boxplot(BS.NLP)
h2 = plot(1:8, mean(BS.NLP),'gs','color','r')
hold off
set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',BS_Methods)
xlabel('');
ylabel('NLP')
set(h1(5,:),'LineWidth',2,'color','k')
set(h1(1,:),'LineWidth',2,'color','k')
set(h1(2,:),'LineWidth',2,'color','k')
set(h1(6,:),'LineWidth',2,'LineStyle',':','color','k')
set(h1(7,~(isnan(h1(7,:)))),'MarkerSize',10,'Color','k','LineWidth',1.5)
set(h2,'MarkerSize',10,'LineWidth',1.5)
print -depsc -tiff -r300 -vector BSBoxNLP.eps
%% 
fontSize=25;
f2 = figure(4)
clf
set(gca,'FontSize',fontSize)
h = glyphplot(BS.RMSE(:,:)','Standardize','matrix','ObsLabels',{'SCtMCMC' ,'SCt4MCMC' , 'tLA' ,'HuberMCMC' ,'HuberLA', 'GP' ,'LaplaceMCMC' ,'LaplaceEP'},'Color','k','Grid', [2, 4])
set(h,'Color','k','LineWidth',1.5)
set(h(:,3),'FontSize',20)
set(f2,'PaperPosition',[0.25 2.5 16 10])
axis off

%% 


f03 = figure(5)
y = [BS.RMSE]; % BS.NLP];
b=bar(y');
% ylim([0 0.37])
 
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(1*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(2).XEndPoints;
ytips1 = b(2).YEndPoints;
labels1 = string(2*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')


xtips1= b(3).XEndPoints;
ytips1 = b(3).YEndPoints;
labels1 = string(3*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(4).XEndPoints;
ytips1 = b(4).YEndPoints;
labels1 = string(4*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(5).XEndPoints;
ytips1 = b(5).YEndPoints;
labels1 = string(5*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(6).XEndPoints;
ytips1 = b(6).YEndPoints;
labels1 = string(6*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(7).XEndPoints;
ytips1 = b(7).YEndPoints;
labels1 = string(7*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
xtips1= b(8).XEndPoints;
ytips1 = b(8).YEndPoints;
labels1 = string(8*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(9).XEndPoints;
ytips1 = b(9).YEndPoints;
labels1 = string(9*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(10).XEndPoints;
ytips1 = b(10).YEndPoints;
labels1 = string(10*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
% ylabel('RMSE')
set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',BS_Methods)



f3 = figure(6)
y = [BS.MAE]; % BS.NLP];
b=bar(y');
ylim([0 0.37])
 
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(1*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
ylabel('MAE')
xtips1= b(2).XEndPoints;
ytips1 = b(2).YEndPoints;
labels1 = string(2*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')


xtips1= b(3).XEndPoints;
ytips1 = b(3).YEndPoints;
labels1 = string(3*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(4).XEndPoints;
ytips1 = b(4).YEndPoints;
labels1 = string(4*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(5).XEndPoints;
ytips1 = b(5).YEndPoints;
labels1 = string(5*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(6).XEndPoints;
ytips1 = b(6).YEndPoints;
labels1 = string(6*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(7).XEndPoints;
ytips1 = b(7).YEndPoints;
labels1 = string(7*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
xtips1= b(8).XEndPoints;
ytips1 = b(8).YEndPoints;
labels1 = string(8*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(9).XEndPoints;
ytips1 = b(9).YEndPoints;
labels1 = string(9*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(10).XEndPoints;
ytips1 = b(10).YEndPoints;
labels1 = string(10*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',BS_Methods)

%% 
f4 = figure(7)
y = [BS.NLP]; % BS.NLP];
b=bar(y');
% ylim([-0.5 0.37])
 
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(1*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
ylabel('NLP')
xtips1= b(2).XEndPoints;
ytips1 = b(2).YEndPoints;
labels1 = string(2*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')


xtips1= b(3).XEndPoints;
ytips1 = b(3).YEndPoints;
labels1 = string(3*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(4).XEndPoints;
ytips1 = b(4).YEndPoints;
labels1 = string(4*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(5).XEndPoints;
ytips1 = b(5).YEndPoints;
labels1 = string(5*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(6).XEndPoints;
ytips1 = b(6).YEndPoints;
labels1 = string(6*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(7).XEndPoints;
ytips1 = b(7).YEndPoints;
labels1 = string(7*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
xtips1= b(8).XEndPoints;
ytips1 = b(8).YEndPoints;
labels1 = string(8*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(9).XEndPoints;
ytips1 = b(9).YEndPoints;
labels1 = string(9*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1= b(10).XEndPoints;
ytips1 = b(10).YEndPoints;
labels1 = string(10*ones(8,1));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',BS_Methods)
set(gca,'color','w','linewidth',1)
set(gca,'xTick',1:8)
set(gca,'xTickLabel',BS_Methods)