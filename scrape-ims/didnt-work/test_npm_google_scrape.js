var Scraper = require('google-images-scraper');
 
var scraper = new Scraper({
    keyword: 'banana',
    rlimit: 10	// 10 p second 
});
 
scraper.list(10).then(function (res) {
    console.log(res);
});