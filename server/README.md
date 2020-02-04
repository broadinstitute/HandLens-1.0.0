# Overview
multipart.js is a server implemented in node.js which enables the SHERLOCK-Reader app to
upload and analyze images. When installed and running, the server listens on port 3000
for push requests. 


# Installation instructions
First, install node.js and npm:

sudo apt update
sudo apt install nodejs	
sudo apt install npm

Next, install and run the server with the following commands:

cd ${PATH_TO_REPOSITORY}/server
sudo npm install express
sudo npm install multer
node multipart.js

The last command installs further dependencies and initiates the server. If the server is
turned off, it can later be restarted simply by typing:

cd ${PATH_TO_REPOSITORY}/server
node multipart.js

Please note that the working directory when the server is started must be the server code
directory.
