# makes list of akc.org dog breed sites for each letter in the alphabet
import string
import pickle as pk

alph = list(string.ascii_uppercase)

urls = []

for lett in alph:
    urls.append('http://www.akc.org/dog-breeds/?letter=' + lett)
    
pk.dump(urls, open('akc-urls.pk', 'wb'))