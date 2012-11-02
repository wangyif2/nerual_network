function [dEdW,dEdV,MSE,MCE] = calcGrad(inputs_train, target_train,W,V)
% [dEdW,dEdV,MSE] = calcGrad(trainPats,W,V)
% BACKPROP CALCULATION OF GRADIENT OF SQUARED ERROR
%   n is the number of input units
%   h is the number of hidden units
%   m is the number of output units
%   N is the number of training cases
%   inputs_train is an n x N matrix of inputs
%   target_train is an m x N matrix of outputs
%   V is h x (n+1), giving the weights to the hidden units and their biases.
%   W is m x (h+1), giving the weights to the output units and their biases.
%
%   dEdW and dEdV are the gradients of the standard squared error function
%          with respect to the weights W and V
%
%   MSE is the normalized value of the error: 0.5 * sum(sum( (Y-D).^2 )) / N
%
%   NB: THIS ASSUMES A ONE HIDDEN LAYER NET WITH SIGMOID UNITS

N  = size(inputs_train, 2); 
n = size(V,2);
h = size(V,1); 
m = size(W,1); 

sumSqrError = 0.0;
sumClassError = 0;
dEdW = zeros(size(W));
dEdV = zeros(size(V));

for pat = 1:N
    %%%%% forward pass %%%%%
    X = [inputs_train(:,pat)',[1]]';
    hidNetIn = V * X;
    hidAct = sigmoid(hidNetIn);    %hidden layer output
    hidActBias = [[hidAct]',[1]]'; %take bias as fixed input
    outNetIn = W * hidActBias;
    outAct = sigmoid(outNetIn);    %outlayer output

    %%%%% backward pass %%%%%
    target = target_train(:, pat);
    error = outAct - target;
    errorClass = ((outAct(:,1) > 0.5) ~= target(:,1));
    sumClassError = sumClassError + errorClass(1,1);
    sumSqrError = sumSqrError + .5*error'*error;
    outDel = 2 * error .* outAct .* (1-outAct);
    dEdW = dEdW + outDel * hidActBias';
    hidDel = (W' * outDel) .* hidActBias .* (1-hidActBias);
    dEdV = dEdV + hidDel(1:h,:) * X';
end

MSE = sumSqrError / N;
MCE = sumClassError / N;
