#!bash

source .env

if [ -e ./data ]; then
  echo "./data exists. You should remove it before re-running."
  exit 1
fi

echo $LIGHTCURVEDB_BACKEND_TYPE
echo $SOCAT_CLIENT_PICKLE_PATH

python3 mock_socat.py
python3 lightserver.py &

sleep 3

curl -X PUT http://localhost:8001/instruments/ \
  -H "Content-Type: application/json" \
  -d '{
        "telescope": "lat",
        "instrument": "latr",
        "frequency": 90,
        "module": "i1",
        "details": {}
      }'


lightcurvedb-socat

sotrp -c config.json

wait