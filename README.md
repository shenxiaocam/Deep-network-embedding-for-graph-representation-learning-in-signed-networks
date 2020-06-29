Code and Datasets for our paper:

**X. Shen and F.-L. Chung, "Deep network embedding for graph representation learning in signed networks," IEEE Transactions on Cybernetics, vol. 50, no. 4, pp. 1556-1568, 2020.**  

<a>

**DNE-SBP Model Descriptions**:
</a>

**Input**:  

Load ".mat" file and get an input matrix **"Gwl_ud"**, i.e., the signed adjacency matrix of a network.

**Hyperparameters**:
1) beta: ratio of penalty on reconstruction errors of observed connections over that of unobserved connections 

2) r: 
   #positive edges / #negative edges; 
   ratio of penalty for reconstruction errors of negative links over that of positive links;
   ratio of weight of pairwise constraints on negatively connected nodes over that of positively connected nodes

3) alfa1: weight of pairwise constraints at 1-st layer of SAE
   alfa2: weight of pairwise constraints at deep layers of SAE

**Output**:  

Low-dimensional node vector representations learned by DNE-SBP are stored in the variable: **rep**  


We test DNE-SBP for link sign prediction & signed network community detection.


**Link sign prediction**  
The function DNESBP_LP() in file “DNESBP_LP.m" can generate low-dimensional node vector representations for link sign prediction
Test examples:

1) In MATLAB, run “DNESBP_LP_wiki.m”, “DNESBP_LP_slashdot.m”, “DNESBP_LP_epinions.m” for example link sign prediction results on Wiki, Slashdot and Epinions datasets, respectively.

2) Use variable “trp” to assign different training percentages. 
   For example, 
   "trp=0.2" indicates training percentage fixed as 20%. 
   "trp=[0.2,0.4,0.6,0.8]" indicates training percentage can be varied among 20%, 40%, 60% and 80%.

3) The AUC and AP averaged over 5 random splits are stored in the variables: “avgAUC” and “avgAPN”, where
   each row corresponds to a type of edge feature, i.e., "L1", "L2", "Had", and "Avg";
   each column corresponds to a specific training percentage, e.g., " 20%", "40%", "60%" or "80%".




**Signed network community detection**  
The function DNESBP_CD() in file “DNESBP_CD.m" can generate low-dimensional node vector representations for signed network community detection
Test examples:

1) In MATLAB, run files “DNESBP_CD_wiki.m”, “DNESBP_CD_slashdot.m”, “DNESBP_CD_epinions.m” for example community detection results on Wiki, Slashdot and Epinions datasets, respectively.

2) Use variable “numCluster” to assign different numbers of clusters. 
   For example, "numCluster=2:10" indicates the number of clusters can be varied between 2 and 10. 

3) The error rates of signed network clustering are stored in the variable: “errorAllK”, where 
   each k-th column corresponds to the error rate given a specific number of k clusters.

