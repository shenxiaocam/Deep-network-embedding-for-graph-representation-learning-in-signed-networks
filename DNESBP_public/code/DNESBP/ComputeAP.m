% pl=load('data.csv');% a table with predicted propabilities and ground-truth labels (1/0)
% % pl=1-pl; %negative class

%% compute average precision
function AP=ComputeAP(pl)
rpl=sortrows(pl,1,'descend'); %rank pl by predicted probabilities in descending order
precision=zeros(size(rpl,1),1);
recall=zeros(size(rpl,1),1);

%% calculate precision and recall at each index from 1 to N (number of testing examples)
for i=1:size(rpl,1)
    precision(i)=sum(rpl(1:i,2))/i;
    recall(i)=sum(rpl(1:i,2))/sum(rpl(:,2));
end
    
rpl=[rpl precision recall];

%% calculate average precision
[Urecall,~,~] = unique(rpl(:,end));
AP=0; %average precision
for i=1:length(Urecall)
    indexR=find(rpl(:,end)==Urecall(i));
    AP=AP+max(rpl(indexR,3)); %find the max precision with the fixed recall
end
AP=AP/size(Urecall,1);

end