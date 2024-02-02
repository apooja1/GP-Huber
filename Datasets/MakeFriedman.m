% Make Friedman data sets
clear all
seed = 0;
randn('state',seed)
rand('state',seed)

nFolds = 10;
N = 100;

for n = 1:nFolds

    % Training data
    XTrain = rand(N,10);
    f = 10 * sin(pi .* XTrain(:,1) .* XTrain(:,2)) + 20 * (XTrain(:,3)-0.5).^2 + 10 * XTrain(:,4) + 5 * XTrain(:,5);
    yTrain = f + randn(size(f));
    outliers = randperm(N);
    outliers = outliers(1:10);
    yTrain(outliers) = 15 + randn(10,1)*3;

    % Standardise
    m = mean(XTrain);
    v = std(XTrain);
    XTrain = (XTrain - repmat(m,N,1))./repmat(v,N,1);
    mY = mean(yTrain);
    vY = std(yTrain);
    yTrain = (yTrain - mY) ./ vY;
    
    % Test data    
    XTest = rand(10000,10);
    fTest = 10 * sin(pi .* XTest(:,1) .* XTest(:,2)) + 20 * (XTest(:,3)-0.5).^2 + 10 * XTest(:,4) + 5 * XTest(:,5);
    XTest = (XTest - repmat(m,size(XTest,1),1))./repmat(v,size(XTest,1),1);
    fTest = (fTest - mY) ./ vY;
    
    filename = sprintf('FriedmanFold%d',n);
    save(filename,'XTrain','yTrain','XTest','fTest')
end
