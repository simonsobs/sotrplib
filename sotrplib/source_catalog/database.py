import pandas as pd
import threading
from filelock import FileLock, Timeout  # Import FileLock and Timeout for file-based locking
from sotrplib.sources.sources import SourceCandidate
import os


class SourceCatalogDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.file_lock = FileLock(f"{db_path}.lock")  # Create a file-based lock
        self._initialize_database()

    def _initialize_database(self):
        try:
            self.df = pd.read_pickle(self.db_path)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=[
                'map_id', 'source_type', 'ra', 'dec', 'err_ra', 'err_dec', 'flux', 'err_flux', 'snr', 'freq', 'ctime', 'arr',
                'sourceID', 'matched_filtered', 'renormalized', 'catalog_crossmatch', 'crossmatch_name', 'fit_type', 'fwhm_a',
                'fwhm_b', 'err_fwhm_a', 'err_fwhm_b', 'orientation', 'kron_flux', 'kron_fluxerr', 'kron_radius', 'ellipticity',
                'elongation', 'fwhm'
            ])

    def add_source(self, source: SourceCandidate, overwrite=False):
        try:
            # Attempt to acquire the lock with a timeout
            with self.file_lock.acquire(timeout=60):  # Timeout after 10 seconds
                with self.lock:  # Use threading lock for thread safety
                    if overwrite:
                        self.df = self.df[~((self.df['map_id'] == source.mapid) & (self.df['sourceID'] == source.sourceID))]
                    new_entry = pd.DataFrame([{
                        'map_id': source.mapid, 'source_type': source.source_type, 'ra': source.ra, 'dec': source.dec,
                        'err_ra': source.err_ra, 'err_dec': source.err_dec, 'flux': source.flux, 'err_flux': source.err_flux,
                        'snr': source.snr, 'freq': source.freq, 'ctime': source.ctime, 'arr': source.arr, 'sourceID': source.sourceID,
                        'matched_filtered': source.matched_filtered, 'renormalized': source.renormalized,
                        'catalog_crossmatch': source.catalog_crossmatch, 'crossmatch_name': source.crossmatch_name,
                        'fit_type': source.fit_type, 'fwhm_a': source.fwhm_a, 'fwhm_b': source.fwhm_b, 'err_fwhm_a': source.err_fwhm_a,
                        'err_fwhm_b': source.err_fwhm_b, 'orientation': source.orientation, 'kron_flux': source.kron_flux,
                        'kron_fluxerr': source.kron_fluxerr, 'kron_radius': source.kron_radius, 'ellipticity': source.ellipticity,
                        'elongation': source.elongation, 'fwhm': source.fwhm
                    }])
                    self.df = pd.concat([self.df, new_entry], ignore_index=True)
                    self.df.to_pickle(self.db_path)
        except Timeout:
            print(f"Could not acquire lock for {self.db_path}.lock. It might be stale.")
            # Optionally, check if the lock file is stale and remove it
            if self._is_lock_stale():
                print("Removing stale lock file.")
                self.file_lock.release(force=True)

    def _is_lock_stale(self):
        """Check if the lock file is stale."""
        lock_file = f"{self.db_path}.lock"
        if os.path.exists(lock_file):
            # Check if the process that created the lock is still running
            try:
                with open(lock_file, "r") as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # Check if the process is alive
            except (ValueError, ProcessLookupError):
                return True  # Lock is stale
        return False

    def observation_exists(self, map_id):
        with self.file_lock:  # Use file-based lock for consistency
            with self.lock:
                return not self.df[self.df['map_id'] == map_id].empty

    def ingest_json(self, json_data, overwrite=False):
        data = json.loads(json_data)
        for entry in data:
            source = SourceCandidate(
                ra=entry['ra'], dec=entry['dec'], err_ra=entry.get('err_ra'), err_dec=entry.get('err_dec'), flux=entry['flux'],
                err_flux=entry.get('err_flux'), snr=entry.get('snr'), freq=entry.get('freq'), ctime=entry.get('ctime'),
                arr=entry.get('arr'), mapid=entry['map_id'], sourceID=entry.get('sourceID'),
                matched_filtered=entry.get('matched_filtered', False), renormalized=entry.get('renormalized', False),
                catalog_crossmatch=entry.get('catalog_crossmatch', False), crossmatch_name=entry.get('crossmatch_name', ''),
                fit_type=entry.get('fit_type', 'forced'), fwhm_a=entry.get('fwhm_a'), fwhm_b=entry.get('fwhm_b'),
                err_fwhm_a=entry.get('err_fwhm_a'), err_fwhm_b=entry.get('err_fwhm_b'), orientation=entry.get('orientation'),
                kron_flux=entry.get('kron_flux'), kron_fluxerr=entry.get('kron_fluxerr'), kron_radius=entry.get('kron_radius'),
                ellipticity=entry.get('ellipticity'), elongation=entry.get('elongation'), fwhm=entry.get('fwhm')
            )
            self.add_source(source, overwrite)


