# Observing clear air vertical motion from space

This repository contains the code in order to produce vertical velocity retrievals from geostationary satellite imagery. The scientific description of the retrieval, as well as its theoretical basis, are described in [REF]. Please refer to this article where the domain of applicability and the limitations of the method are described.

To run, the algorithm requires :
 - Latitude, longitude and Satellite Zenith angle on the geostationary satellite native grid (the algorithm cannot be run at degraded resolution)
 - Brightness temperature measurements at subhourly frequency on that same grid, in the water vapor channel, and the 'clean' and 'dirty' infrared window channels

 <p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"></a></p>  
To provide credit to the authors, please cite Poujol and Bony (in revision).
