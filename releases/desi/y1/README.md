# DESI DR1 BGS Galaxy Group Catalog


## Internal Version Note
This folder contains a (presently) INTERNAL galaxy group catalog built from DESI DR1 Bright Galaxy Survey ('IRON' reductions). Please get in touch with the authors if you are using this for DESI science. It is intended for public release.

Ian Williams (ianwilliams@nyu.edu)
Jeremy Tinker (jlt12@nyu.edu).


## Overview

This catalog is built using the Self Calibrated Group Finder, a public halo-based group finder. The code used for preprocessing and group finding can be found in [SelfCalGroupFinder](https://github.com/ianw89/SelfCalGroupFinder). For details regarding the group finding methods, see Williams *et al.* (2026, in prep). Previous works this catalog builds on include [Tinker (2007)](https://ui.adsabs.harvard.edu/abs/2020arXiv200712200T/abstract) and [Tinker (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...923..154T/abstract). Also see [galaxygroupfinder.net](https://www.galaxygroupfinder.net) for updates and errata.

## Data Access

**Data URL**: <https://data.desi.lbl.gov/public/dr1/vac/dr1/bgs-groups>{: target='_blank'}

NERSC access:
```
/global/cfs/cdirs/desi/public/dr1/vac/dr1/bgs-groups
```

## Documentation

### Catalog Structure

The catalog is provided as a single table and not the two tables of groups and galaxies that a normalization scheme would suggest. This is very simple and avoids and foreign key lookups, but it does mean group properties are duplicated: `M_HALO`, `N_SAT`, and `L_TOT`. As such, to examine group properties, you usually want to first select only the central galaxies to get a single member of each group. For instance, to make a histogram of the group masses in Python:

```python
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

filepath = '/path/to/catalog'
tbl = Table.read(filepath)

df = tbl.to_pandas()

plt.hist(df.loc[~df.IS_SAT, 'M_HALO'], bins=np.logspace(10, 15, 50))
plt.loglog()
```

Note that when converting redshifts to distances we use the DESI fiducial cosmology of [Planck Collaboration (2018)](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P/abstract), a flat $\Lambda$CDM cosmology with $\Omega_m$ = 0.315192, and with $H_0$ = 100 $h$ km/s/Mpc, so all units have factors of $h$ in them as documented below.

### Catalog File

- ``bgs-groups-dr1-v1.0.fits``: The catalog is provided as a single table in the ``GALAXIES`` extension HDU.

### Data Model

Each row in the table is a single galaxy. The columns are:

| Name | Type | Units | Description |
| - | - | - | - |
| TARGETID | int64 | - | DESI's unique target-level identifier, for matching with other DESI data products |
| RA | float64 | degrees | Right Ascension |
| DEC | float64| degrees | Declination |
| Z | float64 | - | Redshift; either as observed or assigned. See Z_ASSIGNED_FLAG. |
| L_GAL | float64 | $L_\odot / h^2$ | Luminosity as converted from r-band absolute Mag (see below). |
| ABS_MAG_R | float64 | magnitudes | Absolute magnitude in r-band, k-corrected to z=0.1, with $H_0$ = 100 km/s/Mpc. |
| VMAX | float64 | Mpc${}^3$ | Maximum volume this galaxy could be observed in based on L_GAL; intended for 1/VMAX corrections for this flux-limited sample (editor: do these units contain $h$?). |
| P_SAT | float64 | - | A number between 0 and 1 indicating how likely this galaxy is to be a satellite as per the group finding algorithm. It is not a true probability; See the corresponding paper for details. When greater than 0.5 the galaxy is considered a satellite, and IS_SAT will be marked True. |
| M_HALO | float64 | $M_\odot / h$ | Group property - assigned halo mass (of entire group). |
| N_SAT | int32 | - | Group property - the number of satellites in the group which this galaxy is part of. |
| L_TOT | float64 | $L_\odot / h^2$ | Group property - the total observed luminosity of the group in which this galaxy is part of. |
| IGRP | int64 | - | Group property - unique group identifier. All members of a group share this number. |
| WEIGHT | float64 | - | Internal detail of group finding. Related to how the luminosity was weighted in abundance matching; see the corresdponding paper for details on the 'w_cen' parameters. |
| APP_MAG_R | float64 | magnitudes | r-band apparent magnitude. |
| Z_ASSIGNED_FLAG | int32 | - | An enumerated value to specify where the redshift is from. 0=DESI spectra. -1=SDSS spectra. -2=random redshift (rare).  3=photometric redshift. n>0 means the redshift was taken from the nth nearest on-sky neighbor, using all observed galaxies in BGS BRIGHT+FAINT as the catalog to draw from. Details on these redshifts assignments can be found in the corresponding paper. |
| G_R | float64 | magnitudes | g-r color on the absolute magnitudes k-corrected to z=0.1. |
| IS_SAT | bool | - | True if this is a satellite; false for centrals. All groups have 1 central. |
| QUIESCENT | bool | - | True if we consider this a quiescent galaxy. |
| LOGMSTAR | float64 | $\log L_\odot / h^2$ | Stellar mass estimated from the [fastspecfit VAC](fastspecfit.md). |

### Randoms

Provided alongside the catalog are randoms files with the same footprint as the group catalog. The format of the randoms files is the same as as the randoms files in the Large-Scale Structure Catalogs with two additional columns. See [here](https://desidatamodel.readthedocs.io/en/latest/DESI_ROOT/survey/catalogs/RELEASE/LSS/SPECPROD/LSScats/VERSIONpip/random_full_pip.html) for the data model of the LSS Catalogs randoms catalogs. The two additional columns are:

| Name | Type | Units | Description |
| - | - | - | - |
| NTILE_ALT | int64 | - | How many DR1 Bright tile centers are within 5862 arcseconds of this point. |
| NTILE_NEARESTIDS |  int64; shape=(15,) | - | Nearest 15 Bright TILEIDs from DR1 to this point. |

NTILE_ALT >= 3 is a requirement to include a galaxy in this group catalog.


### Usage Advice

#### Halo masses

As with all halo-based group finders, the mass estimates should be treated with caution. While a ~0.2 dex error is typical, it can be much larger for certain subpopulations and can easily bias results when utilized in ways that have not been explicitly checked in mock studies such as those of Tinker 2022 (editor: link needed). Galaxies with no satellites and lost galaxy centrals will have much larger uncertainties in the halo mass.

#### Lost galaxies

Our handling of lost galaxies places galaxies at the wrong redshift around 50% of the time. Although we studied several biases induced and have partially ameliorated them, many other biases could still be induced. For some statistics, filtering out this population will be preferable, but this must be evaluated on a case-by-case basis as the removal of this non-random subset also induces biases.

#### High redshift

As with any flux-limited sample, the decreasing target density at high redshift means that observed population are often bright centrals with no observed satellites, triggering the issues in halo mass previously described. Although we include BGS targets to z=0.5, the usefulness of the high-z regime is limited.

#### Spurious objects

Although we have attempted to remove as many non-galactic objects as possible from the catalog, inevitably some objects will avoid our cuts. We advise extra care especially when examining extremal edges of the data: the brightest/faintest galaxies, z > 0.3 and z < 0.01, etc.

#### Rich clusters

The halo model is insufficient to describe unrelaxed systems. Because rich clusters are far more likely to be unrelaxed, our halo-model based description of these systems may be misleading and should be used with caution.

#### Groups on the footprint edge

When a galaxy is near the edge of the footprint, other group members may lie outside our sample and therefore be excluded from the catalog. This also leads to a systematic underestimation of the halo mass assigned to these galaxies. Due to our 3-pass footprint definition and the DR 1 footprint, the geometry of our footprint contains small holes that can trigger this effect.

#### Quiescent vs Star-forming

Classifying quiescent galaxies is difficult at low luminosities, especially fainter than -17 (editor: in what units?). Classifications should be used with caution in this regime.

## Contact

Contact [Ian Williams](mailto:ianwilliams@nyu.edu) or [Jeremy Tinker](mailto:jlt12@nyu.edu) for questions about this catalog.
