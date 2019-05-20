function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = theta*data;
M = bsxfun(@minus, M, max(M, [], 1));
M = bsxfun(@rdivide, exp(M), sum(exp(M)));
M = groundTruth .* M;

M = log(M (M~=0) );

% Decay terms
theta_flat = theta(:);
theta_flat = theta_flat .^ 2;
term2 = (lambda/2)*sum(theta_flat);

cost = -mean(M) + term2;

% Gradient
% TODO below vars are calculated twice save it earlier
sub_max = bsxfun(@minus, theta*data, max(theta*data, [], 1)); % these are calculated twice
temp = bsxfun(@rdivide, exp(sub_max), sum(exp(sub_max)));

gt_term = groundTruth - temp;
decay_term = lambda.*theta;

thetagrad = ((-1/size(data,2))*data*gt_term')' + decay_term;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

