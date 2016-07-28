from urllib.request import build_opener
import simplejson
from io import StringIO

fetcher = build_opener()
searchTerm = 'parrot'
startIndex = 0
searchUrl = "http://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=" + searchTerm + "&start=" + str(startIndex)
f = fetcher.open(searchUrl)
deserialized_output = simplejson.load(f)

print(deserialized_output)