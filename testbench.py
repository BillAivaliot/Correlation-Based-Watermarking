import numpy
import scipy
import cv2
import wm_space

image=cv2.imread("flower.png",0)
#cv2.imwrite("Lena2.jpg",image)
wm=wm_space.generate_watermark(image,20)
i=7678687
wmd_im=wm_space.add_wm_space(image,wm,7678687,1)
cv2.imwrite("flowerwm_space_20.png",wmd_im)

noise=numpy.random.normal(loc=0,scale=70,size=(numpy.shape(wmd_im)))

#print(noise)

wmd_im=cv2.resize(wmd_im,(numpy.uint8(0.5*numpy.shape(wmd_im)[0]),numpy.uint8(0.5*numpy.shape(wmd_im)[1])))
#wmd_im=wmd_im[32:numpy.shape(wmd_im)[0]-13,17:numpy.shape(wmd_im)[1]-16]
#wmd_im=wmd_im+noise
#cv2.imwrite("flowerwmnoisy.png",wmd_im)


f=wm_space.check_watermark(wmd_im,wm,i-1,1,0.1)
print(i-1)
print(f)
f=wm_space.check_watermark(wmd_im,wm,i-2,1,0.1)
print(i-2)
print(f)

f=wm_space.check_watermark(wmd_im,wm,i-3,1,18)
print(i-3)
print(f)
f=wm_space.check_watermark(wmd_im,wm,i-4,1,0.1)
print(i-4)
print(f)
f=wm_space.check_watermark(wmd_im,wm,i-5,1,0.1)
print(i-5)
print(f)
f=wm_space.check_watermark(wmd_im,wm,i,1,0.1)
print(i)
print(f)
f=wm_space.check_watermark(wmd_im,wm,i+1,1,0.1)
print(i+1)
print(f)
f=wm_space.check_watermark(wmd_im,wm,i+2,1,0.1)
print(i+2)
print(f)
f=wm_space.check_watermark(wmd_im,wm,i+3,1,0.1)
print(i+3)
print(f)

#for i in range (7678677, 7678688):
# if (i==7678687):
#  print ("gotit")
#print(wmd_im)
