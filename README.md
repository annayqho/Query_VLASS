The VLA Sky Survey (VLASS; Lacy et al. 2019), has been mapping the entire sky visible to the VLA at low frequencies (2-4 GHz)
in three epochs at a cadence of 32 months.
The Quicklook images are now available for the first two epochs (17,000 square degrees).
This code searches the existing Quicklook data.
It locates the appropriate VLASS tile and subtile for a given RA and Dec
and extracts a cutout 12 arcsec on a side.
Given a non-detection it estimates an upper limit on the flux density by taking the standard deviation of the pixel
values in this cutout, after performing initial 3sigma clipping
(removing pixels with a value greater than 3x the standard deviation).
