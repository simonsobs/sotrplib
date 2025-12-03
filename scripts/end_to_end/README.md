End-to-End Simulation Test
==========================

A full end-to-end simulation test for the SOTRP. This:

1. Generates a synthetic SOCat database and serializes it as a pickle file.
2. Generates an empty lightcurvedb, and runs it in the background as a container.
3. Synchronizes this lightcurvedb with the contents of the SOCat.
4. Starts up a lightserve and lightgest server.
5. Runs the time domain pipeline on a large simulated field with data from SOCat.
6. Shows the output using a local version of lightgest.

To run the pipeline, simply run `run.sh`.

Then, in another terminal; set up the frontend: npm install; export VITE_SERVICE_URL="http://localhost:8000"; npm run dev