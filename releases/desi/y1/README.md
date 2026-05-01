# DESI BGS Y1 Galaxy Group Catalog
Ian Williams (ianwilliams@nyu.edu)
Jeremy Tinker (jlt12@nyu.edu).

This folder contains a (presently) INTERNAL galaxy group catalog built from DESI DR1 Bright Galaxy Survey ('IRON' reductions). Please get in touch with the authors if you are using this for DESI science. It is intended for public release.

The catalog is built using the Self Calibrated Group Finder, a public halo-based group finder. The code used for preprocessing and group finding can be found in [SelfCalGroupFinder](https://github.com/ianw89/SelfCalGroupFinder). For details regarding the group finding methods, see the corresponding paper (link available soon). Previous works this catalog builds on include [arXiv:2007.12200](https://arxiv.org/abs/2007.12200) and [arXiv:2010.02946](https://arxiv.org/abs/2010.02946). Also see [galaxygroupfinder.net](http://www.galaxygroupfinder.net) for updates and errata. 

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
- **L_GAL (float64)**: luminosity in [solar luminosities / h^2] as converted from r-band absolute Mag (see below).
- **ABS_MAG_R (float64)**: absolute magnitude in r-band, k-corrected to z=0.1, with H=100.
- **VMAX (float64)**: max volume this galaxy could be observed in based on L_GAL; intended for 1/VMAX corrections for this flux-limited sample.
- **P_SAT (float64)**: a number between 0 and 1 indicating how likely this galaxy is to be a satellite as per the group finding algorithm. It is not a true probability; See the corresponding paper for details. When greater than 0.5 the galaxy is considered a satellite, and IS_SAT will be marked True.
- **M_HALO (float64)**: group property - assigned halo mass (of entire group) in [solar masses / h]
- **N_SAT (int32)**: group property - the number of satellites in the group which this galaxy is part of.
- **L_TOT (float64)**: group property - the total observed luminosity of the group in which this galaxy is part of [solar luminosities / h^2].
- **IGRP (int64)**: group property - unique group identifier. All members of a group share this number.
- **WEIGHT (float64)**: internal detail of group finding. Related to how the luminosity was weighted in abundance matching; see the corresdponding paper for details on the 'w_cen' parameters.
- **APP_MAG_R (float64)**: r-band apparent magnitude.
- **Z_ASSIGNED_FLAG (int32)**: an enumerated value to specify where the redshift is from. 0=DESI spectra. -1=SDSS spectra. -2=random redshift (rare).  3=photometric redshift. n>0 means the redshift was taken from the nth nearest on-sky neighbor, using all observed galaxies in BGS BRIGHT+FAINT as the catalog to draw from. Details on these redshifts assignments can be found in the corresponding paper.
- **G_R (float64)**: g-r color on the absolute magnitudes k-corrected to z=0.1.
- **IS_SAT (bool)**: True if this is a satellite; false for centrals. All groups have 1 central.
- **QUIESCENT (bool)**: True if we consider this a quiescent galaxy.
- **LOGMSTAR (float64)**: stellar mass estimated from the fastspecfit VAC [log solar masses / h^2].

The catalog is provided as a single table and not the two tables of groups and galaxies that a normalization scheme would suggest. This is very simple and avoids and foreign key lookups, but it does mean group properties are duplicated: `M_HALO`, `N_SAT`, and `L_TOT`. As such, to examine group properties, you usually want to first select only the central galaxies to get a single member of each group. For instance, to make a  histogram of the group masses in Python:

```python
import matplotlib.pyplot as plt
import numpy as np
plt.hist(df.loc[~df.IS_SAT, 'M_HALO'], bins=np.logspace(10, 15, 50))
plt.loglog()
```

Note that when converting redshifts to distances we use the DESI fiducial cosmology of Planck2018, which flat LCDM cosmology with Omega_M=0.315192, but with H=100h with h=1, so all units have factors of little h in them as documented above.

**Usage Advice**


    **Halo masses.** As with all halo-based group finders, the mass estimates should be treated with caution. While a ~0.2 dex error is typical, it can be much larger for certain subpopulations and can easily bias results when utilized in ways that have not been explicitly checked in mock studies such as those of Tinker 2022. Galaxies with no satellites and lost galaxy centrals will have much larger uncertainties in the halo mass.
    **Lost galaxies.** Our handling of lost galaxies places galaxies at the wrong redshift around 50% of the time. Although we studied several biases induced and have partially ameliorated them, many other biases could still be induced. For some statistics, filtering out this population will be preferable, but this must be evaluated on a case-by-case basis as the removal of this non-random subset also induces biases.
    **High redshift.** As with any flux-limited sample, the decreasing target density at high redshift means that observed population are often bright centrals with no observed satellites, triggering the issues in halo mass previously described. Although we include BGS targets to z=0.5, the usefulness of the high-z regime is limited.
    **Spurious objects.** Although we have attempted to remove as many non-galactic objects as possible from the catalog, inevitably some objects will avoid our cuts. We advise extra care especially when examining extremal edges of the data: the brightest/faintest galaxies, z > 0.3 and z < 0.01, etc.
    **Rich clusters.** The halo model is insufficient to describe unrelaxed systems. Because rich clusters are far more likely to be unrelaxed, our halo-model based description of these systems may be misleading and should be used with caution.
    **Groups on the footprint edge.** When a galaxy is near the edge of the footprint, other group members may lie outside our sample and therefore be excluded from the catalog. This also leads to a systematic underestimation of the halo mass assigned to these galaxies. Due to our 3-pass footprint definition and the DR 1 footprint, the geometry of our footprint contains small holes that can trigger this effect.
    **Quiescent vs Star-forming.** Classifying quiescent galaxies is difficult at low luminosities, especially fainter than -17. Classifications should be used with caution in this regime.
    
\end{itemize}