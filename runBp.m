%% To run this program:
%%   First run initBp
%%   Then repeatedly call runBp until convergence.

ErrorsLastRun = zeros(1,NumEpochs);
TestErrorsLastRun = zeros(1,NumEpochs);
startEpoch = totalNumEpochs + 1;
NTest = size(inputs_test,2);
dWold = zeros(size(W));
dVold = zeros(size(V));
reverseStr = '';

for epoch = 1:NumEpochs
  
  %%%%% Calculate the gradient of the objective function %%%  
  [dEdW,dEdV,MSE,MCE] = calcGrad(inputs_train,target_train,W,V);

  %%%%% Update the weights at the end of the epoch %%%%%%
  [W, dW, V, dV] = updateWts(W,dEdW,dWold,V,dEdV,dVold,eta,alpha);
  dWold = dW;   dVold = dV;

  %%%%% Test network's performance on the test patterns %%%%%
  sumSqrTestError = 0;
  sumClassTestError = 0;
  for pat = 1:NTest
      %%%%% forward pass %%%%%
      X = [inputs_test(:,pat)',[1]]';
      hidNetIn = V * X;
      hidAct = sigmoid(hidNetIn);
      hidActBias = [[hidAct]',[1]]';
      outNetIn = W * hidActBias;
      outAct = sigmoid(outNetIn);
      target = target_test(:,pat);
      error = outAct - target;
      errorClass = ((outAct(:,1) > 0.5) ~= target(:,1));
      sumClassTestError = sumClassTestError + errorClass(1,1);
      sumSqrTestError = sumSqrTestError + .5*error'*error;
  end;
   
  %%%%%% Print out summary statistics at the end of the epoch %%%%%
  gradSize = norm([dEdV(:);dEdW(:)]);
  totalNumEpochs = totalNumEpochs + 1;
  TestMSE = sumSqrTestError/NTest;
  TestMCE = sumClassTestError/NTest;
  if totalNumEpochs == 1
      startError = MSE;
  end
  ErrorsLastNumEpochs(1,epoch) = MSE;
  TestErrorsLastNumEpochs(1,epoch) = TestMSE;
  ErrorsClassLastNumEpochs(1,epoch) = MCE;
  TestErrorsClassLastNumEpochs(1,epoch) = TestMCE;

  
  msg = sprintf('%d  MSError=%f, MSTestError=%f, MCError = %f, MCTestError=%f |G|=%f\n',...
            totalNumEpochs,MSE,TestMSE,MCE,TestMCE,gradSize);
  fprintf([reverseStr, msg]);
  reverseStr = repmat(sprintf('\b'), 1, length(msg));
        
end

% clf; 
if totalNumEpochs > minEpochsPerErrorPlot
  epochs = [1:totalNumEpochs];
end

%%%%%%%%% Plot the learning curve for the training set patterns %%%%%%%%%
figure(1);
errorsPerEpoch(1,startEpoch:totalNumEpochs) = ErrorsLastNumEpochs;
subplot(4,1,1), ...
  axis([1 max(minEpochsPerErrorPlot,totalNumEpochs) 0 max(errorsPerEpoch)]),  hold on, ...
  plot(epochs(1,1:totalNumEpochs),errorsPerEpoch(1,1:totalNumEpochs)),...
  title('Mean Squared Error on the Training Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('MSE');


%%%%%%%%% Plot the learning curve for the test set patterns %%%%%%%%%
TestErrorsPerEpoch(1,startEpoch:totalNumEpochs) = TestErrorsLastNumEpochs;
subplot(4,1,2), ...
  axis([1 max(minEpochsPerErrorPlot,totalNumEpochs) 0 max(TestErrorsPerEpoch)]),  hold on, ...
  plot(epochs(1,1:totalNumEpochs),TestErrorsPerEpoch(1,1:totalNumEpochs)), ...
  title('Mean Squared Error on the Test Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('MSE');

%%%%%%%%% Plot the learning curve for the training set patterns with Mean Classification Error %%%%%%%%%
errorsClassPerEpoch(1,startEpoch:totalNumEpochs) = ErrorsClassLastNumEpochs;
subplot(4,1,3), ...
  axis([1 max(minEpochsPerErrorPlot,totalNumEpochs)+0.5 0 max(errorsPerEpoch)+0.5]),  hold on, ...
  plot(epochs(1,1:totalNumEpochs),errorsClassPerEpoch(1,1:totalNumEpochs)),...
  title('Mean Classification Error on the Training Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('MCE');


%%%%%%%%% Plot the learning curve for the test set patterns Mean Classification Error %%%%%%%%%
TestErrorsClassPerEpoch(1,startEpoch:totalNumEpochs) = TestErrorsClassLastNumEpochs;
subplot(4,1,4), ...
  axis([1 max(minEpochsPerErrorPlot,totalNumEpochs)+0.5 0 max(TestErrorsPerEpoch)+0.5]),  hold on, ...
  plot(epochs(1,1:totalNumEpochs),TestErrorsClassPerEpoch(1,1:totalNumEpochs)), ...
  title('Mean Classification Error on the Test Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('MCE');