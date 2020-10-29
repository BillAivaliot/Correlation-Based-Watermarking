import numpy
import scipy
import cv2

def dct2(image_in_all_the_people): #row-column implementation of discreet cosine transform
 dim=numpy.shape(image_in_all_the_people)
 g=numpy.zeros(dim)
 out=numpy.zeros(dim)
 for i in range (0, dim[0]):
  g[i,:]=scipy.fft.dct(image_in_all_the_people[i,:])
 for j in range (0, dim[1]):
  out[:,j]=scipy.fft.dct(g[:,j])
 return out

def idct2(cos_t): #row-column implementation of inverse discreet cosine transform
 dim=numpy.shape(cos_t)
 g=numpy.zeros(dim)
 out=numpy.zeros(dim)
 for i in range (0, dim[0]):
  g[i,:]=scipy.fft.idct(cos_t[i,:])
 for j in range (0, dim[1]):
  out[:,j]=scipy.fft.idct(g[:,j])
 return out

def generate_watermark(image_in_all_the_people,const):
 b=numpy.random.randint(3,size=(numpy.shape(image_in_all_the_people)));
 b=numpy.multiply(b,(b!=2))-(b==2)
 #2 counts as -1
 b=b*const
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

def add_watermark(image_in_all_the_people,watermark,watermark_key,window_size):
 wm=watermark
 dim=numpy.shape(image_in_all_the_people)
 if(dim==numpy.shape(watermark)):
  image_dct=dct2(image_in_all_the_people)
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
  #print(added_wm)
  added_wm_dct=dct2(added_wm)

  #keep the mid frequencies of the watermark
  added_wm_dct_mid=numpy.triu(added_wm_dct)
  wmdif_dct=numpy.zeros(dim)
  wmdif_dct[0:numpy.uint8(numpy.floor(dim[0]/8)),0:numpy.uint8(numpy.floor(dim[1]/8))]=numpy.triu(added_wm_dct[0:numpy.uint8(numpy.floor(dim[0]/8)),0:numpy.uint8(numpy.floor(dim[1]/8))])
  added_wm_dct_mid=added_wm_dct_mid-wmdif_dct
  wmd_dct=image_dct+added_wm_dct_mid #add watermark to the dct of the image
  wmd_image=idct2(wmd_dct) #inverse dct of watermarked dct
  return wmd_image
 else:
  print("image and watermark must have the same dimensions")
  return image_in_all_the_people

def check_watermark(image_in_all_the_people, pattern, watermark_key, block_dim,thres):
 dim=numpy.shape(image_in_all_the_people)
 dimwm=numpy.shape(pattern)
 if(thres>=(-1) and thres<=1):
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

  added_wm_dct=dct2(added_wm)
  
  #keeping mid frequencies of dct
  added_wm_dct_mid=numpy.triu(added_wm_dct)
  wmdif_dct=numpy.zeros(dimwm)
  wmdif_dct[0:numpy.uint8(numpy.floor(dimwm[0]/8)),0:numpy.uint8(numpy.floor(dimwm[1]/8))]=numpy.triu(added_wm_dct[0:numpy.uint8(numpy.floor(dimwm[0]/8)),0:numpy.uint8(numpy.floor(dimwm[1]/8))])
  added_wm_dct_mid=added_wm_dct_mid-wmdif_dct


  if(dimwm==dim): #if the dimensions match we check for the correlation between the dct of the image and the watermark
   wmd_image_dct=dct2(image_in_all_the_people)
   wmd_image_dct=numpy.triu(wmd_image_dct)
   wmd_im_dif=numpy.zeros(dim)
   wmd_im_dif[0:numpy.uint8(numpy.floor(dim[0]/8)),0:numpy.uint8(numpy.floor(dim[1]/8))]=numpy.triu(wmd_image_dct[0:numpy.uint8(numpy.floor(dim[0]/8)),0:numpy.uint8(numpy.floor(dim[1]/8))])
   wmd_image_dct=numpy.subtract(wmd_image_dct,wmd_im_dif)
   imvec=numpy.subtract(numpy.ndarray.flatten(wmd_image_dct),numpy.mean(numpy.ndarray.flatten(wmd_image_dct)))
   wmvec=numpy.subtract(numpy.ndarray.flatten(added_wm_dct_mid),numpy.mean(numpy.ndarray.flatten(added_wm_dct_mid)))
   cor=numpy.inner(numpy.divide(imvec,(numpy.linalg.norm(imvec))),numpy.divide(wmvec,(numpy.linalg.norm(wmvec))))
  if(dimwm>dim): #we might suspect that the image has been produced by croping our copyrighted image
   ext_wmd_image=numpy.zeros(dimwm)
   ext_wmd_image[0:dim[0],0:dim[1]]=image_in_all_the_people #missing pixels are zeros
   wmd_image_dct=dct2(ext_wmd_image)
   wmd_image_dct=numpy.triu(wmd_image_dct)
   wmd_im_dif=numpy.zeros(dimwm)
   wmd_im_dif[0:numpy.uint8(numpy.floor(dimwm[0]/8)),0:numpy.uint8(numpy.floor(dimwm[1]/8))]=numpy.triu(wmd_image_dct[0:numpy.uint8(numpy.floor(dimwm[0]/8)),0:numpy.uint8(numpy.floor(dimwm[1]/8))])
   wmd_image_dct=numpy.subtract(wmd_image_dct,wmd_im_dif)
   imvec=numpy.subtract(numpy.ndarray.flatten(wmd_image_dct),numpy.mean(numpy.ndarray.flatten(wmd_image_dct)))
   wmvec=numpy.subtract(numpy.ndarray.flatten(added_wm_dct_mid),numpy.mean(numpy.ndarray.flatten(added_wm_dct_mid)))
   cor=numpy.inner(numpy.divide(imvec,(numpy.linalg.norm(imvec))),numpy.divide(wmvec,(numpy.linalg.norm(wmvec))))
   if(cor<thres): #we can check if the image has been produced by scalling the watermarked image 

    wm_shrunk=cv2.resize(added_wm,(dim[1],dim[0])) #honey i shrunk the watermark
    wm_shrunk_dct=dct2(wm_shrunk)
    wm_shrunk_dct=numpy.triu(wm_shrunk_dct)
    wm_shrunk_dif=numpy.zeros(dim)
    wm_shrunk_dif[0:numpy.uint8(numpy.floor(dim[0]/8)),0:numpy.uint8(numpy.floor(dim[1]/8))]=numpy.triu(wm_shrunk_dct[0:numpy.uint8(numpy.floor(dim[0]/8)),0:numpy.uint8(numpy.floor(dim[1]/8))])
    wm_shrunk_dct=numpy.subtract(wm_shrunk_dct,wm_shrunk_dif)
    
    scale_im_dct=dct2(image_in_all_the_people)
    
    scale_im_dct=numpy.triu(scale_im_dct)
    wmd_im_dif=numpy.zeros(dim)
    wmd_im_dif[0:numpy.uint8(numpy.floor(dim[0]/8)),0:numpy.uint8(numpy.floor(dim[1]/8))]=numpy.triu(scale_im_dct[0:numpy.uint8(numpy.floor(dim[0]/8)),0:numpy.uint8(numpy.floor(dim[1]/8))])
    wmd_image_dct_sc=numpy.subtract(scale_im_dct,wmd_im_dif)


    
    imvec=numpy.subtract(numpy.ndarray.flatten(wmd_image_dct_sc),numpy.mean(numpy.ndarray.flatten(wmd_image_dct_sc)))
    wmvec=numpy.subtract(numpy.ndarray.flatten(wm_shrunk_dct),numpy.mean(numpy.ndarray.flatten(wm_shrunk_dct)))
    cor=numpy.inner(numpy.divide(imvec,(numpy.linalg.norm(imvec))),numpy.divide(wmvec,(numpy.linalg.norm(wmvec))))
  print(cor)
  return(cor>=thres)
 else:
  print("correlation threshold must be between 1 and -1")
  return False
