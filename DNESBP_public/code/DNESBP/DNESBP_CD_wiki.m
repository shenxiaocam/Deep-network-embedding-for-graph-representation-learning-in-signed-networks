%% An Example Case %%
clear all;
addpath(genpath('../../code'));

load('wiki_UD.mat'); %dataset
G=graph(Gwl_ud);
edgeNum=numedges(G);


%% this is the configuration of stacked autoencoder %%
num_nodes = size(Gwl_ud,1);                    %number of vertex
nnsize = [num_nodes 512 256 128 64];        %layer-wised setting
len = length(nnsize); % number of layers

rand('state',0)
sae = saesetup(nnsize);

for i = 1: len - 1
    sae.ae{i}.activation_function       = 'tanh';  %tanh, tanh_opt ,sigm
    sae.ae{i}.output                    = 'tanh';
    sae.ae{i}.dropoutFraction           = 0;          %  Dropout fraction, only used for fine tuning
    sae.ae{i}.momentum                  = 0;          %  Momentum
    sae.ae{i}.scaling_learningRate      = 0.95;          %  Scaling factor for the learning rate (each epoch)
    sae.ae{i}.nonSparsityPenalty        = 0;          %  0 indicates Non sparsity penalty
    sae.ae{i}.sparsityTarget            = 0.01;       %  Sparsity target
    sae.ae{i}.inputZeroMaskedFraction   = 0.0;        %  Used for Denoising AutoEncoders
    
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

r=floor(length(find(G.Edges{:,2}==1))/length(find(G.Edges{:,2}==-1)));
% #positive edges/ #negative edges
% r is the ratio of penalty for reconstruction errors of negative links over that of positive links
% r is also the ratio of weight of pairwise constraints for negatively connected nodes over that for positively connected nodes

alfa1=16; %weight of pairwise constraints for 1-st layer of SAE
alfa2=1.5; %weight of pairwise constraints for deeper layers of SAE

%% node vector representation learned by DNE-SBP
rep = DNESBP_CD(sae, nnsize,Gwl_ud, beta,r, alfa1,alfa2);


%% k-means clustering on embedding features
errorAllK=[]; %error rates for different number of clusters (k)
for numCluster=2:10 %number of communities
    idx = kmeans(rep{end},numCluster,'Replicates',20,'MaxIter',1000);
    table=tabulate(idx);
    
    %% compute errors
    error=0; %sum of number of positive links between different clusters and negative links within the same cluster
    for  i=1:edgeNum
        [s,t]=findedge(G,i);
        if((Gwl_ud(s,t)<0)&&(idx(s)==idx(t)))
            %negative links within a cluster
            error=error+1;
        else if((Gwl_ud(s,t)>0)&&(idx(s)~=idx(t)))
                % positive links between clusters
                error=error+1;
            end
        end
    end
    error=error/edgeNum;
    errorAllK=[errorAllK error];
end




