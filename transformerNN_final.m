% Loading the datasets
clearvars;close all;clc


daysForward = 1:1:40;
rmse_mat = [];
for dayInd = daysForward
    powerUsageData = readtable('power_usage_2016_to_2020.csv');
    weatherData = readtable('weather_2016_2020_daily.csv');
    
    % Converting dates to datetime format
    powerUsageData.StartDate = datetime(powerUsageData.StartDate, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    weatherData.Date = datetime(weatherData.Date, 'InputFormat', 'yyyy-MM-dd');
    
    % Indeces to exclude
    excludeYears = [2016,2020];
    powerExcludeInd = ismember(year(powerUsageData.StartDate),excludeYears);
    powerUsageData_orig = powerUsageData;
    powerUsageData = powerUsageData(~powerExcludeInd,:);
    
    weatherExcludeInd = ismember(year(weatherData.Date),excludeYears);
    weatherData_orig = weatherData;
    weatherData = weatherData(~weatherExcludeInd,:);
    
    % Aggregating power usage data to daily using manual computation
    powerUsageData.StartDate = dateshift(powerUsageData.StartDate, 'start', 'day');
    uniqueDates = unique(powerUsageData.StartDate);
    totalDailyPowerUsage = zeros(size(uniqueDates));
    for i = 1:length(uniqueDates)
        totalDailyPowerUsage(i) = sum(powerUsageData{powerUsageData.StartDate == uniqueDates(i), 2});
    end
    dailyPowerUsage = table(uniqueDates, totalDailyPowerUsage, 'VariableNames', {'StartDate', 'TotalDailyPowerUsage'});
    
    % Now merging the datasets
    mergedData = outerjoin(dailyPowerUsage, weatherData, 'LeftKeys', 'StartDate', 'RightKeys', 'Date', ...
                           'MergeKeys', true);
    
    % Now handling missing data - removing rows with NaN values
    mergedData = rmmissing(mergedData);
    
    % Feature selection
    features = mergedData(:, {'TotalDailyPowerUsage', 'Temp_avg', 'Press_avg','Dew_avg'});
    
    % Normalizing the features
    mu = mean(features{:,:});
    sig = std(features{:,:});
    features{:,:} = (features{:,:} - mu) ./ sig;
    
    % Shifting the target variable to create the 'previous day power usage' feature
    shiftSize = dayInd;
    
    features.previousDayPowerUsage = [nan(shiftSize,1); features{1:end-shiftSize, 1}];
    features.previousDayTempAvg = [nan(shiftSize,1); features{1:end-shiftSize, 2}];
    features.previousDayPressAvg = [nan(shiftSize,1); features{1:end-shiftSize,3}];
    features.previousDayDewAvg = [nan(shiftSize,1); features{1:end-shiftSize, 4}];
    
    % Removing the first row with NaN
    features(1:shiftSize, :) = [];
    
    % Preparing data for LSTM
    X = features{:, 5:end};
    y = features{:, 1};
    
    % Reshaping data for LSTM
    numObservations = size(X, 1);
    numFeatures = size(X, 2);
    numTimeSteps = 1;
    X = reshape(X', [numFeatures , numObservations]);
    y = reshape(y, [1, numObservations]);
    % Spliting data into training and test sets
    numTimeStepsTrain = floor(0.8 * numObservations);
    XTrain = X(:, 1:numTimeStepsTrain);
    yTrain = y(:, 1:numTimeStepsTrain);
    XTest = X(:, numTimeStepsTrain+1:end);
    yTest = y(:, numTimeStepsTrain+1:end);
    dateTrain = mergedData.StartDate_Date(1:numTimeStepsTrain);
    dateTest = mergedData.StartDate_Date(numTimeStepsTrain+2:end);
    [dateTest,idx] = sort([dateTest],'ascend');
    
    
    % Parameters
    sequenceLength = 80;
    numChannels = 4;
    
    embeddingOutputSize = 4;
    maxPosition = 300;
    
    numHeads = 8;
    numKeyChannels = 8*embeddingOutputSize;
    layers = [ 
        sequenceInputLayer(numChannels,Name="input")
        positionEmbeddingLayer(embeddingOutputSize,maxPosition,PositionDimension="temporal", Name="word-emb");
        additionLayer(2,Name="add")
        selfAttentionLayer(numHeads,numKeyChannels)
        fullyConnectedLayer(1)
        regressionLayer('Name', 'output')];
    
    lgraph = layerGraph(layers);
    
    lgraph = connectLayers(lgraph,"input","add/in2")
    % figure
    % plot(lgraph)
    
    % Specify the options for training
    options = trainingOptions('adam', ...
        'MaxEpochs', 70, ...
        'MiniBatchSize', 8, ...
        'SequenceLength', sequenceLength, ...
        'Plots', 'none', ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.9, ...
        'LearnRateDropPeriod', 10, ...
        'InitialLearnRate', 0.001, ...
        'Verbose', 0);% ...
        % 'Plots', 'training-progress' ...
        %  );
 
    % Train the Transformer model
    net = trainNetwork(XTrain, yTrain, lgraph, options);
    
    % Make predictions on a test sequence
    YPred = predict(net, XTest);
    
    % Calculating RMSE
    rmse = sqrt(mean((YPred(2:end) - yTest(1:end-1)).^2))

    % Calculating R^2
    R2 = 1-sum((yTest(1:end-1)-YPred(2:end)).^2)/sum((yTest-mean(yTest)).^2)
    
    % Plots of the prediction:
    % figure; 
    if dayInd == 1 || dayInd == 2 || dayInd == 3 || dayInd == 4|| dayInd == 5 || dayInd == 7|| dayInd == 14 || dayInd == 20|| dayInd == 30 || dayInd == 40
        nexttile;
        plot(dateTest(dayInd:end-1),yTest(1:end-1));
        hold on 
        plot(dateTest(dayInd:end-1),YPred(2:end));
        title("Test partition " + num2str(dayInd) + " day(s) ahead predictions");
        legend('True values','Estimations')
    end
    rmse_mat(dayInd) = rmse;
end

if(length(daysForward)>1)
    figure
    plot(daysForward,rmse_mat,'k-')
    legend('RMSE values depicting forecasting power', 'Location','southeast')
    xlabel('Days between current date and predicted date')
    ylabel('RMSE')
    title("Model's ability to predict the power consumption")
end

