function [z,out] = costfunction(s,Data) 
 % Read Data Elements
    x=Data.TrainInputs;
    t=Data.TrainTargets;
    x2 = Data.TestInputs;
    t2 = Data.TestTargets;
 % Selected Feature 
    S=find(s~=0);
 % Number of selected Feature
    nf = numel(S);
 % Ratio of Selected Features
    rf=nf/numel(s);
    
 % Selecting Features
    xs =x(:,S); % Train Inputs
    xs2 = x2(:,S); % Test Inputs

 %  Define Initial FIS Structure with FCM Clustering
    opt = genfisOptions('FCMClustering','NumCluster',3);
    %opt = genfisOptions('GridPartition');
    %opt.NumMembershipFunctions = 3;
    %opt.InputMembershipFunctionType = "gaussmf";
    inFIS = genfis(xs,t,opt);
    %opt = genfisOptions('GridPartition');
    %inFIS = genfis(xs,t,opt);
 %  Training ANFIS
    epoch_num = 50;
    TrainData = [xs, t];
    outFIS = anfis(TrainData,inFIS,epoch_num);
% Evaluation Training
    Evaluation.TrainingOutput = evalfis(xs,outFIS);
    Evaluation.TrainingError = Evaluation.TrainingOutput-t;
    Evaluation.TrainingMSE = mean(Evaluation.TrainingError(:).^2);
    Evaluation.TrainingRMSE = sqrt(Evaluation.TrainingMSE);
% Evaluation Testing
    Evaluation.TestingOutput = evalfis(xs2,outFIS);
    Evaluation.TestingError = Evaluation.TestingOutput-t2;
    Evaluation.TestingMSE = mean(Evaluation.TestingError(:).^2);
    Evaluation.TestingRMSE = sqrt(Evaluation.TestingMSE);
% Output 
    out.S = S;
    out.nf=nf;
    out.rf=rf;
    out.MSE=Evaluation.TestingMSE;
    out.RMSE=Evaluation.TestingRMSE;
    z = out.RMSE;
end