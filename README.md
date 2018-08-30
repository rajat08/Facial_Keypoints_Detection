# Facial Keypoint Detection
All of the starting code and resources you'll need to compete this project are in this Github repository. Before you can get started coding, you'll have to make sure that you have all the libraries and dependencies required to support this project.

Note that this project does not require the use of GPU, so this repo does not include instructions for GPU setup.

Local Environment Instructions
Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
git clone https://github.com/cezannec/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
Create (and activate) a new environment, named ai with Python 3.6 and the numpy and pandas packages for data loading and transformation. If prompted to proceed with the install (Proceed [y]/n) type y.

Linux or Mac:
conda create -n ai python=3.6 numpy pandas
source activate ai
Windows:
conda create --name ai python=3.6 numpy pandas
activate ai
At this point your command line should look something like: (ai) <User>:P1_Facial_Keypoints <user>$. The (ai) indicates that your environment has been activated, and you can proceed with further package installations.

Install PyTorch and torchvision.

Linux or Mac:
conda install pytorch torchvision -c pytorch 
Windows:
conda install -c peterjc123 pytorch-cpu
pip install torchvision
Install a few required pip packages, which are specified in the requirements text file (including OpenCV).

pip install -r requirements.txt
