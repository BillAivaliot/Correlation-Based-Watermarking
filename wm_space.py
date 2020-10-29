import numpy as numpy
import numpy.random
import scipy as scipy

import cv2 as cv2

SINGLE_WINDOW=0 #block size: 16*16
DOUBLE_WINDOW=1 #block size: 32*32


def generate_watermark(image_in_all_the_people,const):
 b=numpy.random.randint(3,size=(numpy.shape(image_in_all_the_people)));
 b=numpy.multiply(b,(b!=2))-(b==2)
 #2 counts as -1
 b=b*const
 #rearanging the blocks of the watermark
 return b

def scramble_blocks(mat_in,block_height,block_width, key):
 dim=numpy.shape(mat_in)
 #print(dim)
 blocks_vert=scipy.uint8((dim[0])/block_height) #number of rows of blocks 
 blocks_hor=scipy.uint8((dim[1])/block_width) #number of columns of blocks
 #key is divided in two parts.
 #the most significant digits determine th shift in the rows, and the least significant digits determine shift in rows
 key_sr=scipy.floor(key/10000) #rows shifted_by key_sr
 key_sc=key-(key_sr*10000)
 matrix_of_matrices=numpy.zeros((blocks_vert,blocks_hor,block_height,block_width))
 matrix_of_matrices2=numpy.zeros((blocks_vert,blocks_hor,block_height,block_width))
 matrix_buf=numpy.zeros(dim)
 matrix_out=numpy.zeros(dim)
 for i in range (blocks_vert):
  for j in range (blocks_hor):
   matrix_of_matrices[int((i+j*key_sc)%blocks_vert),int((j+key_sr)%blocks_hor),:,:]=mat_in[i*block_height:int((i+1)*block_height),j*block_width:int((j+1)*block_width)]
 for i in range (blocks_vert):
  for j in range (blocks_hor):
   matrix_buf[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]=matrix_of_matrices[i,j,:,:]
 for j in range (blocks_hor):
  for i in range (blocks_vert):
   matrix_of_matrices2[int((i+key_sc)%blocks_vert),int((j+i*key_sr+7)%blocks_hor),:,:]=matrix_buf[i*block_height:int((i+1)*block_height),j*block_width:int((j+1)*block_width)]
 for i in range (blocks_vert):
  for j in range (blocks_hor):
   matrix_out[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]=matrix_of_matrices2[i,j,:,:]
 return matrix_out

   
def add_wm_space(image_in_all_the_people,watermark,watermark_key,window_size): #adds_watermark in the space domain
 #rearanging the blocks of the watermark
 wm=watermark
 dim=numpy.shape(image_in_all_the_people)
 if(dim==numpy.shape(watermark)):
  added_wm=numpy.zeros(dim)
  if window_size==1:
   if (((dim[0]%32)!=0)and((dim[1]%32)!=0)):
    added_wm[0:(dim[0]-(dim[0]%32)),0:(dim[1]-(dim[1]%32))]=scramble_blocks(wm[0:(dim[0]-(dim[0]%32)),0:(dim[1]-(dim[1]%32))],32,32,watermark_key)
   elif((dim[0]%32!=0)):
    added_wm[0:(dim[0]-(dim[0]%32)),:]=scramble_blocks(wm[0:(dim[0]-(dim[0]%32)),:],32,32,watermark_key)
   elif((dim[1]%32!=0)):
    added_wm[:,0:(dim[1]-(dim[1]%32))]=scramble_blocks(wm[:,0:(dim[1]-(dim[1]%32))],32,32,watermark_key)
   else:
    added_wm=scramble_blocks(wm,32,32,watermark_key)

  else:
   if (((dim[0]%16)!=0)and((dim[1]%16)!=0)):
    added_wm[0:(dim[0]-(dim[0]%16)),0:(dim[1]-(dim[1]%16))]=scramble_blocks(wm[0:(dim[0]-(dim[0]%16)),0:(dim[1]-(dim[1]%16))],16,16,watermark_key)
   elif((dim[0]%16!=0)):
    added_wm[0:(dim[0]-(dim[0]%16)),:]=scramble_blocks(wm[0:(dim[0]-(dim[0]%16)),:],16,16,watermark_key)
   elif((dim[1]%16!=0)):
    added_wm[:,0:(dim[1]-(dim[1]%16))]=scramble_blocks(wm[:,0:(dim[1]-(dim[1]%16))],16,16,watermark_key)
   else:
    added_wm=scramble_blocks(wm,16,16,watermark_key)
  watermarked_image=image_in_all_the_people+added_wm
  return watermarked_image
 else:
  print("watermark and image dimentions must match")
  return image_in_all_the_people

def check_watermark(image_in_all_the_people, pattern, watermark_key, block_dim,thres):

 dim=numpy.shape(image_in_all_the_people)
 dimwm=numpy.shape(pattern)
 if((thres>=-1) and (thres<=1)):
  added_wm=numpy.zeros(dimwm)
  wm=pattern
  if (block_dim==1):
   if (((dimwm[0]%32)!=0)and((dimwm[1]%32)!=0)):
    added_wm[0:(dimwm[0]-(dimwm[0]%32)),0:(dimwm[1]-(dimwm[1]%32))]=scramble_blocks(wm[0:(dimwm[0]-(dimwm[0]%32)),0:(dimwm[1]-(dimwm[1]%32))],32,32,watermark_key)
   elif((dimwm[0]%32!=0)):
    added_wm[0:(dimwm[0]-(dimwm[0]%32)),:]=scramble_blocks(wm[0:(dimwm[0]-(dimwm[0]%32)),:],32,32,watermark_key)
   elif((dim[1]%32!=0)):
    added_wm[:,0:(dimwm[1]-(dimwm[1]%32))]=scramble_blocks(wm[:,0:(dimwm[1]-(dimwm[1]%32))],32,32,watermark_key)
   else:
    added_wm=scramble_blocks(wm,32,32,watermark_key)

  else:
   if (((dimwm[0]%16)!=0)and((dimwm[1]%16)!=0)):
    added_wm[0:(dimwm[0]-(dimwm[0]%16)),0:(dimwm[1]-(dimwm[1]%16))]=scramble_blocks(wm[0:(dimwm[0]-(dimwm[0]%16)),0:(dimwm[1]-(dimwm[1]%16))],16,16,watermark_key)
   elif((dimwm[0]%16!=0)):
    added_wm[0:(dimwm[0]-(dimwm[0]%16)),:]=scramble_blocks(wm[0:(dimwm[0]-(dimwm[0]%16)),:],16,16,watermark_key)
   elif((dim[1]%16!=0)):
    added_wm[:,0:(dimwm[1]-(dimwm[1]%16))]=scramble_blocks(wm[:,0:(dimwm[1]-(dimwm[1]%16))],16,16,watermark_key)
   else:
    added_wm=scramble_blocks(wm,16,16,watermark_key)
  cor=0

  if (block_dim==1):
   ws=32
  else:
   ws=16
  if(dim==numpy.shape(added_wm)):
   imvec=numpy.subtract(numpy.ndarray.flatten(image_in_all_the_people),numpy.mean(numpy.ndarray.flatten(image_in_all_the_people)))
   wmvec=numpy.subtract(numpy.ndarray.flatten(added_wm),numpy.mean(numpy.ndarray.flatten(added_wm)))
   cor=numpy.inner(numpy.divide(imvec,(numpy.linalg.norm(imvec))),numpy.divide(wmvec,(numpy.linalg.norm(wmvec))))
   #print(image_in_all_the_people)
  elif(dim[0]>= numpy.shape(added_wm)[0] or dim[1]>=numpy.shape(added_wm)[1]): #if watermarked image has been scaled up
   su_wm=cv2.resize(added_wm,(dim[0],dim[1]))
   imvec=numpy.subtract(numpy.ndarray.flatten(image_in_all_the_people),numpy.mean(numpy.ndarray.flatten(image_in_all_the_people)))
   wmvec=numpy.subtract(numpy.ndarray.flatten(su_wm),numpy.mean(numpy.ndarray.flatten(su_wm)))
   cor=numpy.inner(numpy.divide(imvec,(numpy.linalg.norm(imvec))),numpy.divide(wmvec,(numpy.linalg.norm(wmvec))))
  elif (dim[0]<= numpy.shape(added_wm)[0] or dim[1]<=numpy.shape(added_wm)[1]): #if image was scaled down or croped
   #first check if the image was scaled down
   sd_wm=cv2.resize(added_wm,(dim[0],dim[1]))
   imvec=numpy.subtract(numpy.ndarray.flatten(image_in_all_the_people),numpy.mean(numpy.ndarray.flatten(image_in_all_the_people)))
   wmvec=numpy.subtract(numpy.ndarray.flatten(sd_wm),numpy.mean(numpy.ndarray.flatten(sd_wm)))
   cor=numpy.inner(numpy.divide(imvec,(numpy.linalg.norm(imvec))),numpy.divide(wmvec,(numpy.linalg.norm(wmvec))))
   #if  the corelation is high enough we can suspect that the initial watermarked image was scaled down
   #if not we check to see if it was cropped
   if (cor<thres):
    found=False
    for i in range(0,dimwm[0]-dim[0]):
     for j in range(0,dimwm[1]-dim[1]):
      wm_part=added_wm[i:i+dim[0],j:j+dim[1]]
      wmvec_part=numpy.subtract(numpy.ndarray.flatten(wm_part),numpy.mean(numpy.ndarray.flatten(wm_part)))
      cor=numpy.inner(numpy.divide(imvec,(numpy.linalg.norm(imvec))),numpy.divide(wmvec_part,(numpy.linalg.norm(wmvec_part))))
      if(cor>=thres):
       #print("got it: ",watermark_key)
       found=True
       break
     if(found==True):
      break

  print (cor)

  return (cor>thres)
 else:
  print("correlation threshold must be between 1 and -1")
  return False
