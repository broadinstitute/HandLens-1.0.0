var express = require("express");
var app = express();
var multer, storage, path, crypto;
multer = require('multer')
path = require('path');
crypto = require('crypto');


var form = "<!DOCTYPE HTML><html><body><center><img src='uploads/sabeti_logo_636x144.png' /></center><center><p></p><hr /><p></p><p></p><form method='post' action='/upload' enctype='multipart/form-data'><input type='file' name='upload'/><input type='submit' /></form><p></p><p></p><hr /></center></body></html>"

app.get('/', function (req, res){
  res.writeHead(200, {'Content-Type': 'text/html' });
  res.end(form);

});

// Include the node file module
var fs = require('fs');

storage = multer.diskStorage({
    destination: './uploads/',
    filename: function(req, file, cb) {
      return crypto.pseudoRandomBytes(16, function(err, raw) {
        if (err) {
          return cb(err);
        }
        return cb(null, "" + (raw.toString('hex')) + (path.extname(file.originalname)));
      });
    }
  });


// Post files
app.post(
  "/upload", // function(req) {console.log(req);},
  multer({
    storage: storage
  }).single('upload'), function(req, res) {
      const { spawn } = require('child_process');
      const pyProg = spawn('python3', ['../scripts/python/strip_detection.py', '--image_file', 'uploads/' + req.file.filename, '--prediction_mode', '--strip_pixels', req.headers.mscalefactor]);
      
      console.log(req.headers.mscalefactor);
      pyProg.stdout.on('data', function(data) {
          console.log(data.toString());
          res.write(data);
	  // res.end()    //res.end('end data - replace with actual data');
      });
      pyProg.stderr.on('data', function(data) {
	                console.log(data.toString());
	                res.write(data);
	                // res.end()    //res.end('end data - replace with actual data');
	            });
      pyProg.on('close', (code) => {
         res.end()
      });
  });

app.get('/uploads/:upload', function (req, res){
  file = req.params.upload;
  console.log(req.params.upload);
  var img = fs.readFileSync(__dirname + "/uploads/" + file);
  res.writeHead(200, {'Content-Type': 'image/png' });
  res.end(img, 'binary');

});

var port_numb = 3001

app.listen(port_numb, function () {
	   console.log('Example app listening on port ' + port_numb);
});
