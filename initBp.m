%% get the digit data
clear all
load fruit_train
load fruit_test

%% initialize the net structure.
numIn = size(inputs_train, 1);

% numHid = 2;
numHid = 5;
% numHid = 10;
% numHid = 30;
% numHid = 100;

numOut = 2;

%%% make random initial weights smaller, and include bias weights
V = 0.1 * (rand(numHid,numIn+1) - ones(numHid,numIn+1) * .5);
W = 0.1 * (rand(numOut,numHid+1) - ones(numOut,numHid+1) * .5);

eta = 0.002;  %% the learning rate 
% eta = 0.01;  %% the learning rate 
% eta = 0.001;  %% the learning rate 
% eta = 0.02;  %% the learning rate 
% eta = 0.05;  %% the learning rate 

alpha = .0;   %% the momentum coefficient

NumEpochs = 5000    ; %% number of learning epochs (number of passes through the
                 %% training set) each time runbp is called.

totalNumEpochs = 0; %% number of learning epochs so far. This is incremented 
                    %% by numEpochs each time runbp is called.

%%% For plotting learning curves:
minEpochsPerErrorPlot = 200;
errorsPerEpoch = zeros(1,minEpochsPerErrorPlot);
TestErrorsPerEpoch = zeros(1,minEpochsPerErrorPlot);

errorsClassPerEpoch = zeros(1,minEpochsPerErrorPlot);
TestErrorsClassPerEpoch = zeros(1,minEpochsPerErrorPlot);

epochs = [1:minEpochsPerErrorPlot];


