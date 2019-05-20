function [nn, L]  = saenntrain(nn, train_x, train_y, opts, beta,r,network, alfa,val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 8 || nargin == 10,'number ofinput arguments must be 8 or 10')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 10
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;
numbatches = floor(numbatches);


assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
LossEpoches=[];
for epoch = 1 : numepochs

    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        % compute the similarity matrix between the samples in each batch
        batch_x_index=kk((l - 1) * batchsize+1: l * batchsize);
        
        simlarMatrix=zeros(batchsize,batchsize);
        for a=1:batchsize
            for b=1:batchsize
                simlarMatrix(a,b)=network(batch_x_index(a),batch_x_index(b)) ;
            end
        end
        S = simlarMatrix;
        
        
        S_P=max(S,0);
        S_N=-min(S,0);
        D_P = diag(sum(S_P,2)); % the degree matrix of S_P
        D_N = diag(sum(S_N,2)); % the degree matrix of S_N
        L_P= D_P- S_P; %laplace matrix of S_P
        L_N= D_N- S_N; %laplace matrix of S_N
        
        laplace =  L_P - r*L_N;
        
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = saeff(nn, batch_x, batch_y,beta,r,laplace, alfa);
        nn = saebp(nn,batch_x,beta,r,laplace, alfa);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    
%     if opts.validation == 1
%         loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
%         str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%     else
%         loss = nneval(nn, loss, train_x, train_y);
%         str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
%     end
    
%     if (mod(epoch, 10) == 0) % update figure after each 10 epoches
%         if ishandle(fhandle)
%             nnupdatefigures(nn, fhandle, loss, opts, epoch);
%         end  
%     end
    
% %     LossEpoches=[LossEpoches mean(L((n-numbatches):(n-1)))];
%     disp(['epoch ' num2str(epoch) '/' num2str(opts.numepochs) '. Mini-batch loss on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    
end

% LossEpoches
end

