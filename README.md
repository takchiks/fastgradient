# fastgradient
# USER MANUAL 

 

To run this adversarial attack requires a GPU server. In order to be able to run one needs to follow the following steps: 

Setup the environment. 


# Install pytorch on the gpu server 

pip install torch==1.6.0 torchvision==0.7.0 

Clone repository from github. 

git clone https://github.com/takchiks/fastgradient 

Download dataset from: 

 https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip  

# Extract zip and copy folder modelnet40_normal_resampled to data folder 

python adversary.py

and generate adversary examples 

 

# To visualize  

cd visualizer 

python show3d_balls.py 

python show3d_balls_perturb.py 

python show3d_balls_test.py 

 

 
