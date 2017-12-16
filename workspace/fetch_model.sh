curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=0B-w3Rmi68HZeMmRnU1ZJdnJDYkk" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > model.tar.gz
tar -xvf model.tar.gz
rm -rf model.tar.gz
