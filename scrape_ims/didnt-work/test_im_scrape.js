var Scraper = require('image-scraper');
var scraper = new Scraper('https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#safe=off&q=dog+breeds&stick=H4sIAAAAAAAAAONgFuLUz9U3SCqxzClSQjC11LKTrfSTMvNz8tMr9VPyc1OLSzKTE0tSU-IT8zJzE3OskopSU1OKHzHGcwu8_HFPWCps0pqT1xgDuIjUKKTGxeaaV5JZUikkw8UrhbBZg0GKmwvB5QEAVcMDHaIAAAA');
 
scraper.scrape(function(image) { 
    image.save();
});