import pandas as pd
import threading
from filelock import FileLock  # Import FileLock and Timeout for file-based locking
from sotrplib.sources.sources import SourceCandidate
import os


class SourceCatalogDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.file_lock = FileLock(f"{db_path}.lock",timeout=1)  # Create a file-based lock
        self._initialize_database()
        
    def _initialize_database(self):    
        self.df = pd.DataFrame(columns=[
            'map_id', 'source_type', 'ra', 'dec', 'err_ra', 'err_dec', 'flux', 'err_flux', 'snr', 'freq', 'ctime', 'arr',
            'sourceID', 'matched_filtered', 'renormalized', 'catalog_crossmatch', 'crossmatch_name', 'fit_type', 'fwhm_a',
            'fwhm_b', 'err_fwhm_a', 'err_fwhm_b', 'orientation', 'kron_flux', 'kron_fluxerr', 'kron_radius', 'ellipticity',
            'elongation', 'fwhm'
        ])

    def read_database(self):
        with self.file_lock:
            # Check if the lock is stale before acquiring it
            if self._is_lock_stale():
                self.file_lock.release()
            try:
                self.df = pd.read_csv(self.db_path)
            except FileNotFoundError:
                self._initialize_database()
        
        
    def add_sources(self, sources: list):
        """Add a list of SourceCandidates to the database."""
        for source in sources:
            new_entry = pd.DataFrame([{'map_id': source.map_id,
                                    'source_type': source.source_type,
                                    'ra': source.ra,
                                    'dec': source.dec,
                                    'err_ra': source.err_ra,
                                    'err_dec': source.err_dec,
                                    'flux': source.flux,
                                    'err_flux': source.err_flux,
                                    'snr': source.snr,
                                    'freq': source.freq,
                                    'ctime': source.ctime,
                                    'arr': source.arr,
                                    'sourceID': source.sourceID,
                                    'matched_filtered': source.matched_filtered,
                                    'renormalized': source.renormalized,
                                    'catalog_crossmatch': source.catalog_crossmatch,
                                    'crossmatch_name': source.crossmatch_name,
                                    'fit_type': source.fit_type,
                                    'fwhm_a': source.fwhm_a,
                                    'fwhm_b': source.fwhm_b,
                                    'err_fwhm_a': source.err_fwhm_a,
                                    'err_fwhm_b': source.err_fwhm_b,
                                    'orientation': source.orientation,
                                    'kron_flux': source.kron_flux,
                                    'kron_fluxerr': source.kron_fluxerr,
                                    'kron_radius': source.kron_radius,
                                    'ellipticity': source.ellipticity,
                                    'elongation': source.elongation,
                                    'fwhm': source.fwhm
                                }])
            self.df = pd.concat([self.df, new_entry], ignore_index=True)
    
    def write_database(self):
        """Write the DataFrame to a pickle file."""
        # Check if the lock is stale before acquiring it
        if self._is_lock_stale():
            self.file_lock.release()  # Release the stale lock
        with self.file_lock:
            file_exists = os.path.exists(self.db_path)
            self.df.to_csv(self.db_path, mode='a', header=not file_exists, index=False)
            

    def _is_lock_stale(self):
        """Check if the lock file is stale."""
        lock_file = self.file_lock.lock_file
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
        return not self.df[self.df['map_id'] == map_id].empty

    def ingest_json(self, json_data, overwrite=False):
        import json
        data = json.loads(json_data)
        for entry in data:
            source = SourceCandidate(ra=entry['ra'],
                                    dec=entry['dec'],
                                    err_ra=entry.get('err_ra'),
                                    err_dec=entry.get('err_dec'),
                                    flux=entry['flux'],
                                    err_flux=entry.get('err_flux'),
                                    snr=entry.get('snr'),
                                    freq=entry.get('freq'),
                                    ctime=entry.get('ctime'),
                                    arr=entry.get('arr'),
                                    map_id=entry['map_id'],
                                    sourceID=entry.get('sourceID'),
                                    matched_filtered=entry.get('matched_filtered', False),
                                    renormalized=entry.get('renormalized', False),
                                    catalog_crossmatch=entry.get('catalog_crossmatch', False),
                                    crossmatch_name=entry.get('crossmatch_name', ''),
                                    fit_type=entry.get('fit_type', 'forced'),
                                    fwhm_a=entry.get('fwhm_a'),
                                    fwhm_b=entry.get('fwhm_b'),
                                    err_fwhm_a=entry.get('err_fwhm_a'),
                                    err_fwhm_b=entry.get('err_fwhm_b'),
                                    orientation=entry.get('orientation'),
                                    kron_flux=entry.get('kron_flux'),
                                    kron_fluxerr=entry.get('kron_fluxerr'),
                                    kron_radius=entry.get('kron_radius'),
                                    ellipticity=entry.get('ellipticity'),
                                    elongation=entry.get('elongation'),
                                    fwhm=entry.get('fwhm')
                                )
            self.add_source(source, overwrite)


