source .env

echo $SOCAT_CLIENT_PICKLE_PATH

python3 mock_socat.py

python3 lightserver.py &

sleep 3

curl -X PUT http://localhost:8001/bands/ \
  -H "Content-Type: application/json" \
  -d '{
        "name": "f090",
        "telescope": "lat",
        "instrument": "latr",
        "frequency": 90
      }'

export LIGHTCURVEDB_POSTGRES_PORT=$(cat port)
echo "LightcurveDB Port set to ${LIGHTCURVEDB_POSTGRES_PORT}"

lightcurvedb-socat

sotrp -c config.json

wait