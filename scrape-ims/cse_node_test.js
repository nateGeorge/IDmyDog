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

var download = function(uri, filename, callback){
  request.head(uri, function(err, res, body){
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
  var srch = 'Bulldog';
  searchIm(cli, srch);
});

var saveIms = function(search, ims, pageNo) {
	for(var i = 0; i <= ims.length-1; i++) {
		console.log(search + '/' + search + String(pageNo) + '-' + String(i) + '.jpg');
		download(ims[i]['url'], search + '/' + search + String(pageNo) + '-' + String(i) + '.jpg', function(){
		  console.log('done');
		});
	}
}

//todo: add recursive function or something to loop through pages asynchronously

var cntr = 0;
var srch = function(client, search, pages, callback) {
	client.search(search, {page: pages[cntr]}).then(function (images) {
		saveIms(search, images, pages[cntr]);	
	});
}

var searchIm = function(client, search) {
	try {
		fs.accessSync(search, fs.F_OK);
		// Do something
		console.log('\'' + search + '\'' + ' dir exists');
	} catch (e) {
		// It isn't accessible
		fs.mkdir(search);
		console.log('made new dir: ' + search);
	}
	
	console.log('searching');
	var pages = _.range(1, 3);
	
	srch(client, search, pages);

	console.log('searched');
}
// paginate results
/*client.search('Steve Angello', {
    page: 2
});

// search for certain size
client.search('Steve Angello', {
    size: 'large'
});*/

