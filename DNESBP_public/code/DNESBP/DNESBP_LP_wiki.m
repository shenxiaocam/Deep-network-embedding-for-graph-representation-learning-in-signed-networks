%% An Example Case %%
clear all;
addpath(genpath('../../code'));

load('wiki_UD.mat'); %dataset
G=graph(Gwl_ud);
edgeNum=numedges(G);
num_nodes=numnodes(G);


%% Randomly sample a fraction of observed links for training for link sign prediction
trp=0.2; %percentage of observed links used for training fixed as 20%
%trp=[0.2,0.4,0.6,0.8]; %percentage of observed links used for training varied in [20%, 40%, 60%,80%]

numRandomSplit=5; %number of random splits for each training percentage

APNAllPer=cell(1,length(trp));
aucAllPer=cell(1,length(trp));
for trpindex=1:length(trp)
    aucAll=[];
    APNAll=[];
    random_state=0;
    for randomSplit=1:numRandomSplit
        disp(['Given ' num2str(trp(trpindex)*100) '% of observed links for training: ' num2str(randomSplit) '-th random split']);
        rng('default');
        rng(random_state);
        edgeindex=[1:1:edgeNum];
        ranindices = edgeindex(randperm(length(edgeindex),fix(length(edgeindex)*trp(trpindex))));
        test_index=setdiff(edgeindex, ranindices);
        random_state=random_state+1;
        
        Gwl_train=zeros(size(Gwl_ud)); %training adjacency matrix given only observed links
        for i=1:length(ranindices)
            [s,t]=findedge(G,ranindices(i));
            Gwl_train(s,t)=Gwl_ud(s,t);
            Gwl_train(t,s)=Gwl_ud(s,t);
        end
        
        
        %% this is the configuration of stacked autoencoder %%
        nnsize = [num_nodes 256 64];        %layer-wised setting
        len = length(nnsize); % number of layers
        rand('state',0)
        sae = saesetup(nnsize);
        
        for i = 1: len - 1
            sae.ae{i}.activation_function       = 'tanh';  %tanh, tanh_opt ,sigm
            sae.ae{i}.output                    = 'tanh';
            sae.ae{i}.dropoutFraction           = 0;          %  Dropout fraction, only used for fine tuning
            sae.ae{i}.momentum                  = 0.0;          %  Momentum
            sae.ae{i}.nonSparsityPenalty        = 0.0;          %  0 indicates Non sparsity penalty
            sae.ae{i}.sparsityTarget            = 0.05;       %  Sparsity target
            sae.ae{i}.inputZeroMaskedFraction   = 0.0;        %  Used for Denoising AutoEncoders
            sae.ae{i}.scaling_learningRate      = 0.95;          %  Scaling factor for the learning rate (each epoch)
            if i==1
                sae.ae{i}.learningRate              = 0.025;
                sae.ae{i}.weightPenaltyL2           = 0.05;       %  L2 regularization
            else
                sae.ae{i}.learningRate              = 0.015;
                sae.ae{i}.weightPenaltyL2           = 0.25;       %  L2 regularization
            end
        end
        
        
       %% hyperparameter settings
        beta=25; %ratio of penalty on reconstruction errors of observed connections over that of unobserved connections
        
        r=floor(length(find(Gwl_train==1))/length(find(Gwl_train==-1)));
        % #positive edges/ #negative edges
        % r is the ratio of penalty for reconstruction errors of negative links over that of positive links
        % r is also the ratio of weight of pairwise constraints for negatively connected nodes over that for positively connected nodes
        
        alfa1=16; %weight of pairwise constraints for 1-st layer SAE
        alfa2=0.4; %weight of pairwise constraints for 2-nd layer SAE
        
        
        %% node vector representation learned by DNE-SBP
        rep = DNESBP_LP(sae, nnsize,Gwl_train, beta,r, alfa1,alfa2);
        
        
        %% build egde representation
        APNAllEdge=[];
        aucAllEdge=[];
        for edgeType=1:4 % 1 for L1-norm; 2 for L2-norm; 3 for hadmard; 4 for average
            inputSize=size(rep{end},2); %number of features for nodes
            edgeRep_train=zeros(length(ranindices),inputSize); % edge representations for training
            edgeLabel_train=zeros(length(ranindices),1);
            for i=1:length(ranindices)
                [s,t]=findedge(G,ranindices(i));
                %get egde representation of training edge (i,j)
                switch edgeType
                    case 1
                        edgeRep_train(i,:)=abs(rep{end}(s,:)-rep{end}(t,:)); %L1-norm
                    case 2
                        edgeRep_train(i,:)=(rep{end}(s,:)-rep{end}(t,:)).^2; %L2-norm
                    case 3
                        edgeRep_train(i,:)=rep{end}(s,:).*rep{end}(t,:); %hadamard
                    case 4
                        edgeRep_train(i,:)=(rep{end}(s,:)+rep{end}(t,:))/2; %average
                end
                edgeLabel_train(i)= Gwl_ud(s,t);
            end
            
            %get egde representation of testing edge (i,j)
            edgeRep_test=zeros(length(test_index),inputSize); % edge representations for testing
            edgeLabel_test=zeros(length(test_index),1);
            for i=1:length(test_index)
                [s,t]=findedge(G,test_index(i));
                %get the egde representation of edge (i,j)
                switch edgeType
                    case 1
                        edgeRep_test(i,:)=abs(rep{end}(s,:)-rep{end}(t,:)); %L1-norm
                    case 2
                        edgeRep_test(i,:)=(rep{end}(s,:)-rep{end}(t,:)).^2; %L2-norm
                    case 3
                        edgeRep_test(i,:)=rep{end}(s,:).*rep{end}(t,:); %hadamard
                    case 4
                        edgeRep_test(i,:)=(rep{end}(s,:)+rep{end}(t,:))/2; %average
                end
                edgeLabel_test(i)= Gwl_ud(s,t);
            end
            
            
            %% logistic regression to predict link labels
            pred=zeros(size(edgeLabel_test));
            edgeLabel_train(edgeLabel_train==-1)=0; %postive link:1; negative link:0
            b = glmfit(edgeRep_train,edgeLabel_train,'binomial','link','logit');
            probability = glmval(b,edgeRep_test, 'logit');
            
            %% compute AUC score
            [~,~,~,AUC] = perfcurve(edgeLabel_test,probability,1) ;
            aucAllEdge=[aucAllEdge;AUC];
            
            %% compute avergae precision of negative links
            edgeLabel_test1=edgeLabel_test;
            edgeLabel_test1(edgeLabel_test1==-1)=0;
            pl=[probability  edgeLabel_test1];
            AP_N=ComputeAP(1-pl);
            APNAllEdge=[APNAllEdge;AP_N];
            
        end
        aucAll=[aucAll aucAllEdge];
        APNAll=[APNAll APNAllEdge];
        
        fprintf('AUC score for 4 types of edge features: \n');
        aucAllEdge
        fprintf('AP for 4 types of edge features: \n');
        APNAllEdge
    end
    aucAllPer{trpindex}=aucAll;
    APNAllPer{trpindex}=APNAll;
    
end

%% average AUC and AP over 5 random splits
avgAUC=zeros(edgeType,length(trp));
avgAPN=zeros(edgeType,length(trp));
for j=1:length(trp)
    avgAUC(:,j)=mean(aucAllPer{j},2);
    avgAPN(:,j)=mean(APNAllPer{j},2);
end



