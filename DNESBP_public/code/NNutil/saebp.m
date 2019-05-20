function nn = saebp(nn,x,beta,r,laplace, alfa)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
 
    n = nn.n;
    m = size(x, 1);
    
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e.* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear'}
            d{n} = - nn.e;
        case 'tanh_opt'
            d{n} = - nn.e.*(1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{n}.^2));
        case 'tanh'
            d{n} = - nn.e.*(1-nn.a{n}.^2);
    end
    
     %% add more penalty to non-zero elements for autoencoder %% 
     if(beta~=1)
        pos_index=find(x>0);
        d{n}(pos_index)= d{n}(pos_index)*beta; % penalty for non-zero input elements (observed links)
        neg_index=find(x<0);
        d{n}(neg_index)= d{n}(neg_index)*(beta*r); % larger penalty for negative links
     end
    
    
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
            case 'tanh'
                d_act =1-nn.a{i}.^2;
                
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
        end
        
        if(nn.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end
        
        

        %pairewise constraints devirations
        if i==2
            switch nn.activation_function 
            case 'sigm'
                d_act_1 = nn.a{i}(:,2:end) .* (1 - nn.a{i}(:,2:end));
            case 'tanh_opt'
                d_act_1 = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}(:,2:end).^2);
            case 'tanh'
                d_act_1 =1-nn.a{i}(:,2:end).^2;
            end
             
            Y=nn.a{i}(:,2:end);
            d{i}=d{i}+[zeros(size(d{i},1),1) (alfa*(laplace+laplace')*Y).*d_act_1]; %pairewise constraints devirations
        end
        
    end
    
 

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end
end
