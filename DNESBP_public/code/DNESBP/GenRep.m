%% Genereate Graph Representations %%

function rep = GenRep(input_data, sae, nnsize)

len = length(nnsize);

nnFF = nnsetup(nnsize);
nnFF.activation_function              = 'tanh' ;
nnFF.output                           = 'tanh' ;

  
num_layers = len - 1;
for i = 1:num_layers
    nnFF.W{i} = sae.ae{i}.W{1};
end

% do FFNN
nnFF.testing = 1;
nnFF = nnff(nnFF, input_data, zeros(size(input_data,1), nnFF.size(end)));


for i=1:size(nnFF.a,2)-1
    nnFF.a{i}(:,1)=[];  %remove the fistr temr, bias
end

rep = nnFF.a(2:end); %return the hidden representation at each layer

end