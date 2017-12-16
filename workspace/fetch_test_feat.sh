curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=0B-w3Rmi68HZeaFNaZ19WTkR1U1E" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > test.tar.gz
tar -xvf test.tar.gz
rm -rf test.tar.gz
