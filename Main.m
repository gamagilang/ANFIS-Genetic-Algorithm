clc;
clear;
% read DataPGE
dataset = xlsread('cleandataPGEnew.csv');
%dataset = tall(dataset);
Inputs = dataset(:,2:11); 
Targets = dataset(:,12);
nSample = size(Inputs,1);
% Train Data
pTrain = 0.7;
nTrain = round(pTrain*nSample);
TrainInputs = Inputs(1:1380,1:10);
TrainTargets = Targets(1:1380);
% Test Data
TestInputs = Inputs(1381:end,1:10);
TestTargets = Targets(1381:end);
Data.TrainInputs = TrainInputs;
Data.TrainTargets = TrainTargets;
Data.TestInputs = TestInputs;
Data.TestTargets = TestTargets;
Data.nf = 10;
    
%% Genetic Algo
CostFunction= @(s) costfunction(s,Data);
nVar = Data.nf;
VarSize=[1 nVar];   % Decision Variables Matrix Size

%% Initialization
MaxIt=10;       % Maximum Number of Iterations

nPop=70;        % Population Size

pc=0.7;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (Parnets)

pm=0.1;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants

mu=0.02;         % Mutation Rate

beta=8;         % Selection Pressure

disp('Initialization ...');

empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.Out=[];

pop=repmat(empty_individual,nPop,1);

for i=1:nPop
    
    % Initialize Position
    pop(i).Position=randi([0 1],VarSize);
    
    % Evaluation
    [pop(i).Cost, pop(i).Out]=CostFunction(pop(i).Position);
    
end

% Sort Population
Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);
pop=pop(SortOrder);

% Store Best Solution
BestSol=pop(1);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

% Store Cost
WorstCost=pop(end).Cost;
%% Main Loop

for it=1:MaxIt
    
    disp(['Starting Iteration ' num2str(it) ' ...']);
    
    P=exp(-beta*Costs/WorstCost);
    P=P/sum(P);
    
    % Crossover
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
        
        % Select Parents Indices
        i1=RouletteWheelSelection(P);
        i2=RouletteWheelSelection(P);

        % Select Parents
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        
        % Evaluate Offsprings
        [popc(k,1).Cost, popc(k,1).Out]=CostFunction(popc(k,1).Position);
        [popc(k,2).Cost, popc(k,2).Out]=CostFunction(popc(k,2).Position);
        
    end
    popc=popc(:);
    
    
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        
        % Apply Mutation
        popm(k).Position=Mutate(p.Position,mu);
        
        % Evaluate Mutant
        [popm(k).Cost, popm(k).Out]=CostFunction(popm(k).Position);
        
    end
    
    % Create Merged Population
    pop=[pop;popc;popm]; %#ok
     
    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    
    % Update Worst Cost
    WorstCost=max(WorstCost,pop(end).Cost);
    
    % Truncation
    pop=pop(1:nPop);
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=pop(1);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end

%% Results

figure;
plot(BestCost,'-or');
ylabel('Cost (RMSE)');
xlabel('Generation');
grid on
%% Evaluate Best Solution to ANFIS
% Selected Feature 
    Z=find(pop(1).Position~=0);
 % Number of selected Feature
    nf = numel(Z);
 % Ratio of Selected Features
    %rf=nf/numel(s);
    
 % Selecting Features
    x=Data.TrainInputs;
    t=Data.TrainTargets;
    x2 = Data.TestInputs;
    t2 = Data.TestTargets;
    
    xs =x(:,Z); % Train Inputs
    xs2 = x2(:,Z); % Test Inputs
%%
 %  Define Initial FIS Structure with FCM Clustering
    opt = genfisOptions('FCMClustering','NumCluster',3);
    %opt = genfisOptions('GridPartition');
    %opt.NumMembershipFunctions = 4;
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
%%
plot(Data.TrainTargets,'-ro')
hold on
plot(Evaluation.TrainingOutput,'-bo')
legend('Training Target','ANFIS-GA estimation')
xlabel('Data Training')
ylabel('SSC')
grid on
%%
plot(TestTargets,'-ro') 
hold on 
plot(Evaluation.TestingOutput,'-bo')
legend('Test Target','ANFIS-GA estimation')
xlabel('Data Testing')
ylabel('SSC')
    %%
    figure;
    plotfis(inFIS);
%%
 %% FIS Parameter
%showrule(inFIS)
plotmf(inFIS,'input',1);
xlabel('Membership Function for Input 1')
%%
plotmf(inFIS,'input',2);
xlabel('Membership Function for Input 2')
%%
plotmf(inFIS,'input',3);
xlabel('Membership Function for Input 3')
%%
plotmf(inFIS,'input',4);
xlabel('Membership Function for Input 4')
%%
plotmf(inFIS,'input',5);
xlabel('Membership Function for Input 5')
%%
plotmf(inFIS,'input',6);
xlabel('Membership Function for Input 6')
%%
plotmf(inFIS,'input',7);
xlabel('Membership Function for Input 7')
%%
plotmf(inFIS,'input',8);
xlabel('Membership Function for Input 8')
%%
showrule(inFIS)