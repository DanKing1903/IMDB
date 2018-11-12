from pathlib import Path
import urllib.request

def download_if_missing(url, filename):
    path = Path(filename).resolve()
    print("destination: {}".format(path))
    if not path.exists():
        print("downloading file")
        urllib.request.urlretrieve(url, filename)
    else:
        print("file found")

    return filename



if __name__ == '__main__':

    url = 'https://upload.wikimedia.org/wikipedia/commons/0/0b/Cat_poster_1.jpg'
    filename = 'data/Cat_poster_1.jpg'
    download_if_missing(url, filename)
