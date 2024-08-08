# EST-RVI
(1) Experimental Environment:

        scipy>=0.19.0
        numpy>=1.12.1
        pandas>=0.19.2
        tensorflow>=1.3.0
        networkx>=2.5.1
  

(2) DataSets

“Q-traffic” 
This dataset originates from (https://github.com/JingqingZ/BaiduTraffic)
In this paper, the geographic coordinates range of road network is [116.10, 39.69, 116.71, 40.18]. 
It contains 45,148 roadway segments, each counted from April 1 to May 31, 2017 for the three categories. 
This data selected 329 of these vulnerability-prone road segments futher.

“Q-traffic Events” 
The online crowd queries which are derived from users’ navigation search queries can characterize the potential occurrence of events at specific locations. The traffic conditions and crowd queries for these road segments are sampled every 15 minutes with a vector of dimensions
329 × 5856. Additionally, adverse weather events such as windstorms and rainfall are also matched under this dataset.



