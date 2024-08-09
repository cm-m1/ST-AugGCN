# EST-RVI
(1) Experimental Environment:

graphviz                  0.8.4         
grpcio                    1.62.2        
h5py                      3.8.0         
idna                      3.7           
importlib-metadata        6.7.0         
joblib                    1.3.2         
Keras-Applications        1.0.8         
Keras-Preprocessing       1.1.2         
Markdown                  3.4.4         
MarkupSafe                2.1.5         
mock                      5.1.0         
mxnet-cu112               1.9.1         
networkx                  2.6.3         
numpy                     1.21.6        
nvidia-cublas-cu11        11.10.3.66    
nvidia-cuda-nvrtc-cu11    11.7.99       
nvidia-cuda-runtime-cu11  11.7.99       
nvidia-cudnn-cu11         8.5.0.96      
pandas                    1.3.5         
Pillow                    9.5.0         
python                    3.7.12        
scikit-learn              1.0.2         
scipy                     1.7.3         
tensorflow                1.15.0        
tensorflow-estimator      1.15.1        
termcolor                 2.3.0         
threadpoolctl             3.1.0         
tqdm                      4.66.4        

  
(2) DataSets

“Q-traffic” 
This dataset originates from (https://github.com/JingqingZ/BaiduTraffic)
In this paper, the geographic coordinates range of road network is [116.10, 39.69, 116.71, 40.18]. 
It contains 45,148 roadway segments, each counted from April 1 to May 31, 2017 for the three categories. 
This data selected 329 of these vulnerability-prone road segments futher.

“Q-traffic Events” 
The online crowd queries which are derived from users’ navigation search queries can characterize the potential occurrence of events at specific locations. The traffic conditions and crowd queries for these road segments are sampled every 15 minutes with a vector of dimensions
329 × 5856. Additionally, adverse weather events such as windstorms and rainfall are also matched under this dataset.



