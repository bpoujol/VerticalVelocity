# Observing clear air vertical motion from space

This repository contains the code in order to produce vertical velocity retrievals from geostationary satellite imagery. The scientific description of the retrieval, as well as its theoretical basis, are described in [REF]. Please refer to this article where the domain of applicability and the limitations of the method are described.

To run, the algorithm requires :
 - Latitude, longitude and Satellite Zenith angle on the geostationary satellite native grid (the algorithm cannot be run at degraded resolution)
 - Brightness temperature measurements at subhourly frequency on that same grid, in the water vapor channel, and the 'clean' and 'dirty' infrared window channels

The algorithm is licensed under a Creative Commons CC-BY4.0 License. To provide credit to the authors, please cite [REF].
