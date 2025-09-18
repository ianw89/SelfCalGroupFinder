import pickle
import pandas as pd
from typing import Optional, Dict, Tuple
from dataloc import *

class FootprintManager:
    """
    Manages BGS survey footprint calculations from randoms files.
    Caches results in memory and cleans up randoms data after computation.
    Singleton class to ensure only one instance exists.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FootprintManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._footprint_cache: Dict[Tuple[str, str, int], float] = {}
        self.RANDOMS_DENSITY = 2500  # targets per square degree
        self._cache_fn = OUTPUT_FOLDER + 'footprint_cache.pkl'
        
        # File mappings for different surveys
        self._randoms_files = {
            'SV3-18': MY_RANDOMS_SV3_MINI,
            'SV3-20': MY_RANDOMS_SV3_MINI_20,
            'Y1': RANDOMS_Y1_0_WITHMYNTILE, 
            'Y3': RANDOMS_Y3_0_WITHMYNTILE
        }

        # Load cache if it exists
        try:
            with open(self._cache_fn, 'rb') as f:
                self._footprint_cache = pickle.load(f)
        except FileNotFoundError:
            # If cache file does not exist, start with an empty cache
            self._footprint_cache = {}
        
        self._initialized = True
    
    def get_footprint(self, survey: str, region: str = 'all', min_passes: int = 1) -> float:
        """
        Get the footprint area in square degrees for a BGS survey.
        
        Parameters:
        -----------
        survey : str
            Survey name ('SV3-18', 'SV3-20', 'Y1', 'Y3')
        region : str
            Region to calculate ('all', 'N', 'S')
        min_passes : int
            Minimum number of passes (1-10)
            
        Returns:
        --------
        float
            Footprint area in square degrees
        """
        # Validate inputs
        if survey not in self._randoms_files:
            raise ValueError(f"Survey must be one of {list(self._randoms_files.keys())}")
        if region not in ['all', 'N', 'S']:
            raise ValueError("Region must be 'all', 'N', or 'S'")
        if not 1 <= min_passes <= 10:
            raise ValueError("min_passes must be between 1 and 10")
        
        # Check cache first
        cache_key = (survey, region, min_passes)
        if cache_key in self._footprint_cache:
            return self._footprint_cache[cache_key]
        
        # Load and process randoms file
        randoms_file = self._randoms_files[survey]
        randoms_df = pickle.load(open(randoms_file, 'rb'))
        
        try:
            # Calculate footprint for this survey/region/passes combination
            self._calculate_all_footprints(randoms_df, survey)
            
        finally:
            # Clean up randoms data from memory
            del randoms_df
        
        # Return the requested footprint
        return self._footprint_cache[cache_key]
    
    def _calculate_all_footprints(self, randoms_df: pd.DataFrame, survey: str) -> None:
        """
        Calculate and cache all footprint combinations for a survey.
        This minimizes file I/O by computing everything at once.
        """
        for min_passes in range(1, 11):
            # Filter by minimum passes
            pass_filter = randoms_df['NTILE_MINE'] >= min_passes
            randoms_filtered = randoms_df.loc[pass_filter]
            
            # Calculate footprint for 'all'
            footprint_all = len(randoms_filtered) / self.RANDOMS_DENSITY
            self._footprint_cache[(survey, 'all', min_passes)] = footprint_all
            
            # Calculate footprint for North ('N')
            randoms_north = randoms_filtered.loc[randoms_filtered['PHOTSYS'] == 'N']
            footprint_north = len(randoms_north) / self.RANDOMS_DENSITY
            self._footprint_cache[(survey, 'N', min_passes)] = footprint_north
            
            # Calculate footprint for South ('S') 
            randoms_south = randoms_filtered.loc[randoms_filtered['PHOTSYS'] == 'S']
            footprint_south = len(randoms_south) / self.RANDOMS_DENSITY
            self._footprint_cache[(survey, 'S', min_passes)] = footprint_south
    
    def reload_all(self):
        """
        Force reload of all footprints from randoms files.
        This is useful if the randoms files have changed.
        """
        for survey in self._randoms_files.keys():
            self.get_footprint(survey, 'all', 1)

        # Save off the cache to a pickle file
        with open(self._cache_fn, 'wb') as f:
            pickle.dump(self._footprint_cache, f)
    