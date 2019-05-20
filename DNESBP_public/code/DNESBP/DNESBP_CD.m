%% Learn node vector representations by DNE-SBP for signed network community detection %%

%% hyperparameters
%beta:  ratio of penalty on reconstruction errors of observed connections over that of unobserved connections
% r = #positive edges/ #negative edges
% r is the ratio of penalty for reconstruction errors of negative links over that of positive links
% r is also the ratio of weight of pairwise constraints for negatively connected nodes over that for positively connected nodes
%alfa1: weight of pairwise constraints for 1-st layer of SAE
%alfa2: weight of pairwise constraints for deep layers of SAE
%nnsize: Dimensionality of each layer of SAE

%% Inputs
%sae: configuration of stacked autoencoder
%adj: adjacency matrix

%% Output
%rep: node vector representation learned by DNE-SBP for signed network community detection

function rep = DNESBP_CD(sae, nnsize,adj, beta,r, alfa1,alfa2)

sae = saetrain_CD(sae, adj, beta,r,adj, alfa1,alfa2);
rep = GenRep(adj, sae, nnsize);  % node vector representation learned by DNE-SBP

end


function sae = saetrain_CD(sae, x, beta,r,network, alfa1,alfa2)

for i = 1 : numel(sae.ae)
    disp(['Training SAE ' num2str(i) '/' num2str(numel(sae.ae))]);
    opts.batchsize = 1000; %process how many instances in each batch , community detection
    
    if i==1
        opts.numepochs = 100; 
        alfa=alfa1;
    else
        opts.numepochs =50;
      
        beta=1; % not beta penalty on second layer autoencoder
        r=1; % not extra penalty on second layer autoencoder
        alfa=alfa2;
    end
    
    
    sae.ae{i} = saenntrain(sae.ae{i}, x, x, opts,beta,r,network, alfa);
    t = nnff(sae.ae{i}, x, x);

    x = t.a{2};
    %remove bias term
    x = x(:,2:end);
end

end
