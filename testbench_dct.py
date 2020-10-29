import numpy
import scipy
import cv2
import wm_dct
import wm_dct_s
a=cv2.imread("flower.png",0)
wm=wm_dct.generate_watermark(a,10)
noise=numpy.random.normal(loc=0,scale=0,size=(numpy.shape(a)))

wmd_image=wm_dct.add_watermark(a,wm,77686745,0)+noise
wmd2=wmd_image[0:200,20:210]
wmd3=cv2.resize(wmd_image,(numpy.uint8(0.5*numpy.shape(wmd_image)[0]),numpy.uint8(0.5*numpy.shape(wmd_image)[1])))

cv2.imwrite("flowe_watermarked_dct.png",wmd_image)

cv2.imwrite("difference_dct.png",wmd_image-a)



print("START testing WMD1")
wm_found=wm_dct.check_watermark(wmd_image,wm,77686745,0,0.15)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd_image,wm,77686746,0,0.15)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd_image,wm,77686747,0,0.15)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd_image,wm,77686744,0,0.15)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd_image,wm,77686743,0,0.15)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd_image,wm,77686748,0,0.15)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd_image,wm,77686749,0,0.15)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd_image,wm,77686740,0,0.15)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd_image,wm,77686715,0,0.15)
print(wm_found)

print("FINISHED testing WMD1")


print("START testing WMD2")
#wm_found=wm_dct_s.check_watermark(wmd2,wm,77686745,0,0.1)
#print(wm_found)

#wm_found=wm_dct_s.check_watermark(wmd2,wm,77686746,0,0.1)
#print(wm_found)

#wm_found=wm_dct_s.check_watermark(wmd2,wm,77686747,0,0.1)
#print(wm_found)

#wm_found=wm_dct_s.check_watermark(wmd2,wm,77686744,0,0.1)
#print(wm_found)

#wm_found=wm_dct.check_watermark(wmd2,wm,77686743,0,0.1)
#print(wm_found)

#wm_found=wm_dct.check_watermark(wmd2,wm,77686748,0,0.1)
#print(wm_found)

#wm_found=wm_dct.check_watermark(wmd2,wm,77686749,0,0.1)
#print(wm_found)

#wm_found=wm_dct.check_watermark(wmd2,wm,77686740,0,0.1)
#print(wm_found)

#wm_found=wm_dct.check_watermark(wmd2,wm,77686715,0,0.1)
#print(wm_found)

print("FINISHED testing WMD2")

print("START testing WMD3")
wm_found=wm_dct.check_watermark(wmd3,wm,77686745,0,0.1)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd3,wm,77686746,0,0.1)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd3,wm,77686747,0,0.1)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd3,wm,77686744,0,0.1)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd3,wm,77686743,0,0.1)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd3,wm,77686748,0,0.1)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd3,wm,77686749,0,0.1)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd3,wm,77686740,0,0.1)
print(wm_found)

wm_found=wm_dct.check_watermark(wmd3,wm,77686715,0,0.1)
print(wm_found)

print("FINISHED testing WMD3")

#cv2.imwrite("flowerdif.png",c-a)
#cv2.imwrite("flowerdifdct.png",wm_dct.dct2(c-a))
