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