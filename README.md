# CorrelationBasedWatermarking
leanguage: python
additional libraries: numpy, scipy, opencv

Adding and detecting watermark to a black and white image on space and dct domains.

Watermark is a pseudo-random pattern whose dimentions are equal to those of the image.
It is divided to blocks of 16x16 px or 32x32 px.
Each block is matched to a block of the image depending on the key. 

Watermark detection algorithm checks if the image is watermarked or if we can suspect that it has been produced by cropping or scaling a watermarked image.

wm_space.py: Contains functions for adding and detecting watermarks on the space domain
wm_dct_s.py: Contains functions for adding and detecting watermarks on the dct domain.
wm_dct.py: Same as above but wne the watermark detectro checks for cropping, it only checks upper left part of image. A lot faster but less accurate than wm_dct_s.
