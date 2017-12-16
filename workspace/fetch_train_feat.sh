curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=0B-w3Rmi68HZeRk1xU0N6dWFzdFU" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > train.tar.gz
tar -xvf train.tar.gz
rm -rf train.tar.gz
