function nn = saeff(nn, x, y,beta,r,laplace, alfa)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)
%% add more penalty to non-zero elements for autoencoder %%

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;

    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh'
                nn.a{i} = tanh(nn.a{i - 1} * nn.W{i - 1}');
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'tanh_opt'
            nn.a{n} = tanh_opt(nn.a{n - 1} * nn.W{n - 1}'); 
        case 'tanh'
            nn.a{n} = tanh(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
    nn.e = y - nn.a{n};
    
    %% add more penalty to non-zero elements for autoencoder %%    
    if(beta~=1)
        pos_index=find(y>0);
        nn.e(pos_index)= nn.e(pos_index)*beta; % penalty for non-zero input elements (observed links)
        neg_index=find(y<0);
        nn.e(neg_index)= nn.e(neg_index)*(beta*r); % larger penalty for negative links
    end

        
    
    switch nn.output
        case {'sigm', 'linear','tanh_opt','tanh'}  
            nn.L = 1/2 * sum(sum((nn.e).^ 2)) / m;  
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;    
    end
    
    
     %% add pairwise constraints %% 
    Y=nn.a{2}(:,2:end); %hidden representation learned from autoencoder
    nn.L= nn.L+ (alfa/m)*trace(Y'*laplace*Y);
end
