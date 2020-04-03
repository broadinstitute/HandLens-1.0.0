# Overview
multipart.js is a server implemented in node.js which enables the Handlens app to
upload and analyze images. When installed and running, the server listens on port 3000
for push requests. 


# Installation instructions
First, install node.js and npm:
```
sudo apt update
sudo apt install nodejs	
sudo apt install npm
```

Next, install the relevant python libraies:
```
sudo apt install pip3-python
sudo apt install libopencv-dev
pip3 install opencv-python
pip3 install np_utils
pip3 install imutils
pip3 install matplotlib
pip3 install pandas
pip3 install scipy

```

Next, install and run the server with the following commands:
```
cd ${PATH_TO_REPOSITORY}/server
sudo npm install express
sudo npm install multer
node multipart.js
```

The last command installs further dependencies and initiates the server. If the server is
turned off, it can later be restarted simply by typing:
```
cd ${PATH_TO_REPOSITORY}/server
node multipart.js
```

Please note that the working directory when the server is started must be the server code
directory.
