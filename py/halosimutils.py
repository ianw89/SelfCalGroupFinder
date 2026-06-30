import numpy as np
import pandas as pd
from scipy.special import erf

def downsample_halos(df: pd.DataFrame) -> pd.DataFrame:
    exp_cutoff = 12.0 # 10.8
    sigma = 0.6 # 0.4
    def downsample_prob(log_mass, cutoff=exp_cutoff, sigma=sigma):
        return 0.5 * (1 + erf((log_mass - cutoff) / (sigma * np.sqrt(2))))
    random_values = np.random.rand(len(df))

    df_down = df.loc[random_values < downsample_prob(df['LOGMHALO'])]

    return df_down