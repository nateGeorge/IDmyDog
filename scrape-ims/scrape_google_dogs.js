// write to local server: http://stackoverflow.com/questions/5294470/writing-image-to-local-server
// google-images: https://github.com/vdemedes/google-images
// download image from url: http://stackoverflow.com/questions/12740659/downloading-images-with-node-js
// check if dir exists: http://stackoverflow.com/questions/4482686/check-synchronously-if-file-directory-exists-in-node-js
// read file: https://docs.nodejitsu.com/articles/file-system/how-to-read-files-in-nodejs/

'use strict'

const googleImages = require('google-images');
var fs = require('fs');
var request = require('request');
var _ = require('underscore');
var path = require('path');

function getDirectories(srcpath) {
  return fs.readdirSync(srcpath).filter(function(file) {
    return fs.statSync(path.join(srcpath, file)).isDirectory();
  });
}

var dirs = getDirectories('/media/nate/Windows/github/IDmyDog/scrape-ims/images')

dirs.pop('full')

var download = function(uri, filename, callback){
  request.head(uri, function(err, res, body){
    if (err) {
		return console.log(err);
	}
	console.log('content-type:', res.headers['content-type']);
    console.log('content-length:', res.headers['content-length']);

    request(uri).pipe(fs.createWriteStream(filename)).on('close', callback);
  });
};

fs.readFile('credentials.cred', 'utf8', function (err, data) {
  if (err) {
    return console.log(err);
  }
  var lines = data.split('\n');
  var cseID = lines[0];
  var apiKey = lines[1];
  let cli = googleImages(cseID, apiKey);
  //var srch = 'Bulldog';
  var test = dirs.slice(6)
  test.forEach(function(e,i,a) {
	//setTimeout(console.log(e), i*1000);
  	setTimeout(function() {searchIm(cli, e)}, i*25000);
  });
});

var saveIms = function(search, ims, pageNo) {
	for(var i = 0; i <= ims.length-1; i++) {
		download(ims[i]['url'], 'images/' + search + '/' + search + String(pageNo) + '-' + String(i) + '.jpg', function(){
		  console.log('done');
		});
	}
}

var srch = function(client, search, page) {
	console.log('searching for: ' + search + ' breed');
	client.search(search + ' breed', {page: page}).then(function (images) {
		saveIms(search, images, page);	
	});
}

var searchIm = function(client, search) {
	try {
		fs.accessSync('images/' + search, fs.F_OK);
		console.log('\'' + search + '\'' + ' dir exists');
	} catch (e) {
		// It isn't accessible
		fs.mkdir('images/' + search);
		console.log('made new dir: ' + 'images/' + search);
	}
	
	var pages = _.range(1, 5);
	pages.forEach(function(e,i,a) {srch(client, search, e)});
}