// from here: http://stackoverflow.com/questions/18112204/get-all-directories-within-directory-nodejs

var fs = require('fs'),
    path = require('path');

function getDirectories(srcpath) {
  return fs.readdirSync(srcpath).filter(function(file) {
    return fs.statSync(path.join(srcpath, file)).isDirectory();
  });
}

var dirs = getDirectories('/media/nate/Windows/github/IDmyDog/scrape-ims/images')

dirs.pop('full')

console.log(dirs)

console.log(dirs.slice(3))

// from here: http://stackoverflow.com/questions/17614123/node-js-how-to-write-an-array-to-file
// need to keep track of which folders have been used to scrape google images
var file = fs.createWriteStream('google_processed_dirs.txt');
file.on('error', function(err) { console.log(err); });
dirs.forEach(function(v) { file.write(v + '\n'); });
file.end();