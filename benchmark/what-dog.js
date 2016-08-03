var whatDog = require('what-dog');

whatDog('http://imgur.com/B7a15F5.jpg')
    .then(doggyData => {
        console.log(doggyData);
    })

// currently not working
var fs = require('fs');
var http = require('http');

fs.readFile('/home/nate/Downloads/ah1.jpg', function(err, data) {
    if (err) throw err;
    whatDog(data)
	.then(doggyData => {
		console.log('downloaded file');
		console.log(doggyData);
	});
})