# DESI BGS Y3 Galaxy Group Catalog
Ian Williams (ianwilliams@nyu.edu)
Jeremy Tinker (jlt12@nyu.edu).

This folder contains a (presently) INTERNAL galaxy group catalog built from DESI Y3 Bright Galaxy Survey ('LOA' reductions). Please get in touch with the authors if you are using this for DESI science.

The catalog is built using the Self Calibrated Group Finder, a public halo-based group finder. The code used for preprocessing and group finding can be found in [SelfCalGroupFinder](https://github.com/ianw89/SelfCalGroupFinder). For details regarding the group finding methods, see [arXiv:2007.12200](https://arxiv.org/abs/2007.12200), [arXiv:2010.02946](https://arxiv.org/abs/2010.02946) and [galaxygroupfinder.net](http://www.galaxygroupfinder.net), which contains information not yet published describing details of this catalog's construction. This version fo the catalog used group finding parameters that were tuned using SV3 to SDSS clustering and Lsat data.

The catalog is provided as a single table in a .FITS archive. In Python, this can be easily read as follows:

```python
from astropy.table import Table
filepath = '/path/to/catalog'
tbl = Table.read(filepath)

# Optionally convert to a pandas Dataframe
df = tbl.to_pandas()
```

Each row in the table is a single galaxy. The columns are:

- **TARGETID (int64)**: DESI's unique target-level identifier, for matching with other DESI data products.
- **RA (float64)**: right ascension in [degrees].
- **DEC (float64)**: declination in [degrees].
- **Z (float64)**: redshift; either as observed or assigned. See Z_ASSIGNED_FLAG.
- **L_GAL (float64)**: luminosity in [solar luminosities / h^2] as converted from r-band absolute Mag.
- **VMAX (float64)**: max volume this galaxy could be observed in based on L_GAL; intended for 1/VMAX corrections for this flux-limited sample.
- **P_SAT (float64)**: a number between 0 and 1 indicating how likely this galaxy is to be a satellite as per the group finding algorithm. It is not a true probability; See the Tinker papers for details. When greater than 0.5 the galaxy is considered a satellite, and IS_SAT will be marked True.
- **M_HALO (float64)**: group property - assigned halo mass (of entire group) in [solar masses / h]
- **N_SAT (int32)**: group property - the number of satellites in the group which this galaxy is part of.
- **L_TOT (float64)**: group property - the total observed luminosity of the group in which this galaxy is part of [solar luminosities / h^2].
- **IGRP (int64)**: group property - unique group identifier. All members of a group share this number.
- **WEIGHT (float64)**: internal detail of group finding. Related to how the luminosity was weighted in abundance matching; see Tinker 2020 on the 'Chi' parameters.
- **APP_MAG_R (float64)**: r-band apparent magnitude.
- **Z_ASSIGNED_FLAG (int32)**: an enumerated value to specify where the redshift is from. 0=DESI spectra. -1=SDSS spectra. -2=random redshift (rare). -3=photometric redshift. n>0 means the redshift was taken from the nth nearest on-sky neighbor, using all observed galaxies in BGS BRIGHT+FAINT as the catalog to draw from. Details on these redshifts assignments can be found on www.galaxygroupfinder.net.
- **G_R (float64)**: g-r color from the Legacy survey fluxes, k-corrected to z=0.1.
- **IS_SAT (bool)**: True if this is a satellite; false for centrals. All groups have 1 central.
- **QUIESCENT (bool)**: True if we consider this a quiescent galaxy.
- **MSTAR (float64)**: stellar mass estimated from the fastspecfit VAC [solar masses].

The catalog is provided as a single table and not the two tables of groups and galaxies that a normalization scheme would suggest. This is very simple and avoids and foreign key lookups, but it does mean group properties are duplicated: `M_HALO`, `N_SAT`, and `L_TOT`. As such, to examine group properties, you usually want to first select only the central galaxies to get a single member of each group. For instance, to make a  histogram of the group masses in Python:

```python
import matplotlib.pyplot as plt
plt.hist(df.loc[~df.IS_SAT, 'M_HALO'], bins=np.logspace(11, 15, 50))
plt.loglog()
```

Note that when converting redshifts to distances we use flat LCDM cosmology with Omega_M=0.25 and H=100.

**KNOWN ISSUES**

1. There is some contamination of non-galaxy objects in the catalog. This is primarilly from the unobserved targets (no DESI spectra), which would have been filtered out due to a poor spectral fitting. 
