#for bioid or lfpw library, there are three parts need to modify
import multiprocessing
import numpy as np
import time
import cv2
from initialize_face import detect_face
from initialize_face import detect_eye
#from gredient_base import gogo
from align_shape import similarity_transform as st
from align_shape import project_shape as ps
#from sklearn import linear_model
path="params/"
begin=0
shape=np.load(path+"shape(1520,40).npy")
#align_shape=np.load(path+"align_shape1(1520,40).npy")
align_shape0=np.load(path+"align_shape(1520,40).npy")
mean_shape0=np.mean(align_shape0[0:500,:],axis=0) 
mean_shape1=np.expand_dims(mean_shape0,axis=0)
mean_shape0=np.mean(align_shape0[400:410,:],axis=0) 
mean_shape2=np.expand_dims(mean_shape0,axis=0)
mean_shape0=np.mean(align_shape0[300:310,:],axis=0) 
mean_shape3=np.expand_dims(mean_shape0,axis=0)
mean_shape0=np.mean(align_shape0[200:210,:],axis=0) 
mean_shape4=np.expand_dims(mean_shape0,axis=0)
mean_shape0=np.mean(align_shape0[490:500,:],axis=0) 
mean_shape5=np.expand_dims(mean_shape0,axis=0)

compare=np.loadtxt(path+"pupil_by_shape6.txt")

#af=np.load("affine_tran.npy")
fit_shape=np.load(path+"fit_shape(Tr,K,2_F,40).npy")

th=np.load(path+"thresholds(Tr,K,F).npy")
bounding_box=np.load(path+"bound_box(1520,8).npy")
p=28
augmentation=20
total=583
Tmax=10
t=0#train image index for max=1520
K=500
Tr=10
F=5
P=300#ramdomly select points from interest region
beta=1000
batch=583
bins=np.array([[1,1,1,1,1],[1,0,0,0,0],[1,1,0,0,0],[0,1,0,0,0],[1,1,1,0,0],[0,1,1,0,0],[1,0,1,0,0],[0,0,1,0,0],
[1,0,0,1,0],[1,1,0,1,0],[0,1,0,1,0],[1,1,1,1,0],[0,1,1,1,0],[1,0,1,1,0],[0,0,1,1,0],[0,0,0,1,0],
[1,0,0,0,1],[1,1,0,0,1],[0,1,0,0,1],[1,1,1,0,1],[0,1,1,0,1],[1,0,1,0,1],[0,0,1,0,1],
[1,0,0,1,1],[1,1,0,1,1],[0,1,0,1,1],[0,1,1,1,1],[1,0,1,1,1],[0,0,1,1,1],[0,0,0,1,1],
[0,0,0,0,1]])
points=np.zeros((1521,5),dtype=np.int16)
pupil=np.zeros((1521,5))
def get_feature(coor_feature,gray,shape0):
    points=np.zeros((1,P))
    features=np.zeros((P,P),dtype=np.int16)
    for i in range(0,P):
        near=coor_feature[2,i]
        
        yy1,xx1=coor_feature[0,i]+shape0[0,near*2],coor_feature[1,i]+shape0[0,1+near*2]
        #print yy1,xx1
        if  (xx1>gray.shape[1]): 
                xx1=gray.shape[1]-1
                #print i_face,xx1,yy1,xx2,yy2
        if  (yy1>gray.shape[0]): 
                yy1=gray.shape[0]-1
        points[0,i]=gray[yy1,xx1]
    
    for i in range(0,P):
        for j in range(0,P):
            features[i,j]=float(points[0,i])-float(points[0,j])
    return features
    
    
def apply_F_feature(F_feature,coor_feature,gray,b,new_shape):
    #print "apply_F_feature"
    #mean_shape=np.array([[ 108.75920868,162.52566528,  109.39736938 , 225.70263672 , 175.12303162,
    #168.89341736,  175.51907349,  218.61051941, 141.26644897, 193.76118469]])
    #i_face=1
    #cv2.namedWindow("i")
    
    features=np.zeros((1,5))
    
    
    #img[new_shape[0,i*2]:new_shape[0,i*2]+5,new_shape[0,i*2+1]:new_shape[0,i*2+1]+5]=[0,255,0]
    #cv2.imshow("i",img)
    #cv2.waitKey(0)
    #print new_shape
        #print tran,angel,scale,af[i_face,1]/af[i_face,0],af[i_face,1],af[i_face,0]
    for i in range(0,F):
        
            p1,p2=F_feature[i,0],F_feature[i,1]
            near1,near2=coor_feature[2,p1],coor_feature[2,p2]
            x1,y1=coor_feature[0,p1],coor_feature[1,p1]
            x2,y2=coor_feature[0,p2],coor_feature[1,p2]
            xx1,yy1=int(x1+new_shape[0,near1*2]),int(y1+new_shape[0,1+near1*2])
            xx2,yy2=int(x2+new_shape[0,near2*2]),int(y2+new_shape[0,1+near2*2])
            #print new_shape[0,2*p1/n],new_shape[0,2*p1/n+1,],new_shape[0,2*p2/n],new_shape[0,2*p2/n+1]
            #img[new_shape[0,int(p1)/n*2]:new_shape[0,int(p1)/n*2]+5,new_shape[0,1+int(p1)/n*2]:new_shape[0,1+int(p1)/n*2]+5]=[0,255,0]
            #img[new_shape[0,int(p2)/n*2]:new_shape[0,int(p2)/n*2]+5,new_shape[0,1+int(p2)/n*2]:new_shape[0,1+int(p2)/n*2]+5]=[0,255,0]
            if  (xx1>gray.shape[0]-1): 
                xx1=gray.shape[0]-1
                #print i_face,xx1,yy1,xx2,yy2
            if  (yy1>gray.shape[1]-1): 
                yy1=gray.shape[1]-1
                #print i_face,xx1,yy1,xx2,yy2
            if  (xx2>gray.shape[0]-1): 
                xx2=gray.shape[0]-1
                #print i_face,xx1,yy1,xx2,yy2
            if  (yy2>gray.shape[1]-1): 
                yy2=gray.shape[1]-1
                #print i_face,xx1,yy1,xx2,yy2    
            features[0,i]=float(gray[xx1,yy1])-float(gray[xx2,yy2]) 
            
            #print x1,y1,x2,y2,2*int(p1)/n,2*int(p2)/n,p1,p2
            #img=cv2.imread("face_detect/build/BioID-FaceDatabase-V1.2/BioID_"+str(i_face).rjust(4,'0')+".pgm")
            """
            gray[xx1:xx1+2,yy1:yy1+2]=255
            gray[xx2:xx2+2,yy2:yy2+2]=255
            print xx1, yy1, xx2, yy2
            #print p1,p2
    cv2.imshow("i",gray)
    cv2.waitKey(0)"""
    return features
       
def apply_Rt_regression(mean_shape,Rt,coor_feature,gray,features):
    #mean_shape0=np.mean(align_shape,axis=0) 
    #shape0=np.expand_dims(mean_shape0,axis=0)
    F_feature=np.load(path+"F_feature_for_regression"+str(Rt)+"(K,F,2).npy")
    thresholds=np.zeros((1,F))
    #mean_shape,af=vectorize_shape(project_shape1)
    new_shape=np.zeros((1,p*2))
    #mean_shape=mean_shape0    
    #features=get_feature(coor_feature[i,:,:],filename,shape0)
    
    for i in range(0,K):
        
        thresholds[0,:]=0.2*(np.max(features[:,:])-np.min(features[:,:]))*th[Rt,i,:]    
        #thresholds[0,j]=features[th[Rt,i,0,j],th[Rt,i,1,j]]
        
        feature_for_bin=apply_F_feature(F_feature[i,:,:],coor_feature,gray,0,mean_shape)
        signature=np.greater(feature_for_bin[0,:],thresholds[:,:])
        #print signature.shape
        class_bin=sum([signature[0,k]*(2**k) for k in range(0,F)])
        
        
        
        new_shape=(-fit_shape[Rt,i,class_bin,:])+new_shape
    #new_shape=new_shape+mean_shape
    """for k in range(0,p):
            new_xy=np.array([[new_shape[0,k*2]-af[2]],[new_shape[0,k*2+1]-af[3]]])
        
            angel=np.arctan(af[1]/af[0])
            scale=1.00000/(af[0]/np.cos(angel))
            tran=np.array([[scale*np.cos(-angel),-scale*np.sin(-angel)],
            [scale*np.sin(-angel),scale*np.cos(-angel)]])
            new_shape[0,k*2:k*2+2]=np.dot(tran,new_xy).T"""
        
    #print new_shape[0,0],i
        #print mean_shape[0,0],i,fit_shape[Rt,i,class_bin,0]
        
        #print mean_shape
        #mean_shape=np.dot(w_bins[i,:,:],feature_for_bin.T).T+mean_shape
        #print mean_shape,i,Rt
    #cv2.waitKey(0)
    
    return new_shape
def apply_regression(img_copy,mean_shape,mean_x,mean_y,sx,sy,q):
    
    coor_feature0=np.load(path+"coor_feature_initial(Tr,3,P).npy")
    #coor_feature[:,0,:]=np.round(coor_feature[:,0,:]*sizex)
    #coor_feature[:,1,:]=np.round(coor_feature[:,1,:]*sizey)
    img=img_copy.copy()
    gray=cv2.cvtColor(img,cv2.cv.CV_RGB2GRAY)
    features=gray[mean_x-sx:mean_x+sx, mean_y-sy:mean_y+sy]
    sy=features.shape[1]
    sx=features.shape[0]
    rand_y=np.random.randint(sy-1, size=(1,P))
    rand_x=np.random.randint(sx-1, size=(1,P))
    
    mask=np.zeros((sx, sy), dtype=np.int16)
    for i in range(0, P):            
      mask[rand_y[0, i], rand_x[0, i]]=1
    features=features[:,:]*mask
    #bounding_box=np.array([0, 0,mean_y ,sy , 0, 0, mean_x, sx])
    for i in range(0,Tr):
        #print i
        #print mean_shape,"mean_shape"
        #features=get_feature(coor_feature[i,:,:],gray,mean_shape)
        new_coor=np.zeros((3,P),dtype=np.int16)
        coor_feature=coor_feature0[i, :, :]
        #new_shape=np.zeros((1, p*2))
    
    	  	
        #new_shape[0,:]=ps(esti_shape[i,j,:],bounding_box[i,:],-1)
        tran=st(mean_shape[0,:],mean_shape1[0,:])
        temp_xy=np.zeros((1,P*2))

        for k in range(0,P): 
                  temp_xy[0,k*2]=coor_feature[0,k]
                  temp_xy[0,k*2+1]=coor_feature[1,k]
                  temp_xy[0,k*2:k*2+2]=np.dot(tran,temp_xy[0,k*2:k*2+2]).T   
                  new_coor[0,k]=temp_xy[0,k*2]
                  new_coor[1,k]=temp_xy[0,k*2+1]
                  new_coor[2,k]=coor_feature[2,k]
        new_shape=apply_Rt_regression(mean_shape,i,new_coor[:,:],gray, features)
        #print new_shape,"wenew_shape"
        tran=st(mean_shape[0,:],mean_shape1[0,:])
       
        for k in range(0,p):        
                new_shape[0,k*2:k*2+2]=np.dot(tran,new_shape[0,k*2:k*2+2]).T 
        
        mean_shape=mean_shape+new_shape
        #print new_shape
        #print new_shape
        """img=cv2.imread(filename)
        for j in range(0,p):        
            img[int(mean_shape[0,j*2]):int(mean_shape[0,j*2])+2,int(mean_shape[0,j*2+1]):int(mean_shape[0,j*2+1])+2]=[50*i,50*i,0]
            cv2.namedWindow("imshow3")
            cv2.imshow("imshow3",img)
        cv2.waitKey(0)"""
    
    q.put(mean_shape[:,:])
    
    return mean_shape
def get_shape(shape,sizex,sizey,mean_y,mean_x):
    shape0=np.zeros((1,p*2))
    new_shape=np.zeros((1,p*2))
    shape0[:,:]=shape
    shape0[0,::2]=shape0[0,::2]*sizex
    shape0[0,1::2]=shape0[0,1::2]*sizey
    #for lfpw
    delta_x=shape0[0,40]-mean_x
    delta_y=shape0[0,41]-mean_y
        
    #for bioid
    #delta_x=shape0[0,28]-mean_x-0.18*sizex
    #delta_y=shape0[0,29]-mean_y
    #print delta_x,delta_y,shape0[0,28],shape0[0,29],mean_x,mean_y
    #new_shape[0,::2]=shape0[0,::2]-delta_x+0.2*sizex
    #new_shape[0,1::2]=shape0[0,1::2]-delta_y
    new_shape[0,::2]=shape0[0,::2]+mean_x
    new_shape[0,1::2]=shape0[0,1::2]+mean_y
    #new_shape[0,28*2]=new_shape[0,28*2]+4
    #for i in range(18,22):
        #new_shape[0,i*2]=new_shape[0,i*2]-2
    return new_shape
def go(index,show,input_img=0):
    
    global pupil
    #mean_shape0=np.mean(align_shape0,axis=0) 
    #shape0=np.expand_dims(mean_shape0,axis=0)
    new_shape=np.zeros((1,p*2))
    
   
    if (index>=0):
    	
    	filename="/home/wcs/opencv_assignments/BioID-FaceDatabase-V1.2/BioID_"+str(index).rjust(4,'0')+".pgm"
    	img=cv2.imread(filename)
    else:
    	img=input_img
    img_copy=img.copy()
    mean_y,mean_x,sy,sx=detect_face(img)
    #for lfpw
    sizex,sizey=int(sx*0.5),int(sy*0.50)
    
    #for bioid
    #sizex,sizey=sx*0.37,sy*0.38
    if (mean_y!=0):

        #print mean_shape.shape
       
        #cv2.namedWindow("imshow")
        #cv2.imshow("imshow",img)
        #cv2.waitKey(0)
    
        result_shape_output=np.zeros((5,p*2))
        new_shape=np.zeros((5,p*2))
        result_shape=np.zeros((5,p*2))
        new_shape[0,:]=get_shape(mean_shape1[0,:],sizex,sizey,mean_y,mean_x)
        """
        new_shape[1,:]=get_shape(mean_shape2[0,:],sizex,sizey,mean_y,mean_x)
        new_shape[2,:]=get_shape(mean_shape3[0,:],sizex,sizey,mean_y,mean_x)
        new_shape[3,:]=get_shape(mean_shape4[0,:],sizex,sizey,mean_y,mean_x)
        new_shape[4,:]=get_shape(mean_shape5[0,:],sizex,sizey,mean_y,mean_x)     
        new_shape[:,::2]=new_shape[:,::2]-5
        """
        new_shape[1:,:]=new_shape[0,:]
        new_shape[1,::2]=new_shape[1,::2]+8
        new_shape[2,::2]=new_shape[2,::2]-8
        new_shape[3,1::2]=new_shape[3,1::2]+8
        new_shape[4,1::2]=new_shape[4,1::2]-8
        new_shape[:,::2]=new_shape[:,::2]-4

        m_shape=np.mean(new_shape,axis=0)
        
        for i in range(0,p):        
            img[int(m_shape[i*2]):int(m_shape[i*2])+3,int(m_shape[i*2+1]):int(m_shape[i*2+1])+3]=[0,250,0]
        cv2.imshow("wefes",img)
        #cv2.waitKey(0)
        jobs=[]
        q=multiprocessing.Queue()
        for ii in range(0,5):
            #pool=multiprocessing.Pool(process=5)
            #pool.map(apply_regression,filename,new_shape[ii:ii+1,:],sizex,sizey,result_shape)
            #pro=multiprocessing.Process(target=work,args=(ii,))
            #kkk=apply_regression(filename,new_shape[ii:ii+1,:],mean_x,mean_y,sizex,sizey, q)
            #apply_regression(filename,new_shape[ii:ii+1,:],mean_x,mean_y,sizex,sizey, q)
            pro=multiprocessing.Process(target=apply_regression,args=(img_copy,new_shape[ii:ii+1,:],mean_x,mean_y,sizex,sizey, q))
            
            jobs.append(pro)
        
            pro.start()
            #time.sleep(5)
            #a,b=apply_regression(filename,new_shape[ii:ii+1,:],sizex,sizey,q)
            #print result_shape_output[:,:]
            #print result_shape_output[ii,:]
        for kk in range(0,5):
            result_shape_output[kk,:]=q.get()
            #print np.mean(result_shape_output[kk,:])
        #jobs[kk].terminate()
        #jobs[kk].join()
            
        result_shape[:,:]=result_shape_output[:,:]
        #print result_shape    
        result_shape5=np.mean(result_shape,axis=0)
        result_shape5=np.expand_dims(result_shape5,axis=0)
        #print mean_shape
        img=img_copy.copy()
        img2=img_copy.copy()
        img3=img_copy.copy()
        point1x,point1y=new_shape[0,4*2],new_shape[0,4*2+1]
        if (index>=0):
        	#for lfpw
        	pupil[index-begin,0],pupil[index-begin,1]=np.round(result_shape5[0,32]),np.round(result_shape5[0,33])
        	pupil[index-begin,2],pupil[index-begin,3]=np.round(result_shape5[0,34]),np.round(result_shape5[0,35])
        	#for bioid
        	#pupil[index-begin,0],pupil[index-begin,1]=np.round(result_shape5[0,0]),np.round(result_shape5[0,1])
        	#pupil[index-begin,2],pupil[index-begin,3]=np.round(result_shape5[0,2]),np.round(result_shape5[0,3])
        sizex,sizey=new_shape[0,10*2]-new_shape[0,4*2],new_shape[0,10*2+1]-new_shape[0,4*2+1]
        new_img=img[point1x:point1x+3*sizex,point1y:point1y+sizey*1.6,:]
        new_img3=new_img.copy()
        #ddy,ddx=gogo(-1,11,new_img3,new_shape[0,0]-point1x,new_shape[0,1]-point1y)
        temp_p=p
        for i in range(0,temp_p):        
            img2[int(result_shape[0,i*2]):int(result_shape[0,i*2])+2,int(result_shape[0,i*2+1]):int(result_shape[0,i*2+1])+2]=[0,255,0]
        for i in range(0,temp_p):        
            img2[int(result_shape[1,i*2]):int(result_shape[1,i*2])+2,int(result_shape[1,i*2+1]):int(result_shape[1,i*2+1])+2]=[0,255,255]
        for i in range(0,temp_p):        
            img2[int(result_shape[2,i*2]):int(result_shape[2,i*2])+2,int(result_shape[2,i*2+1]):int(result_shape[2,i*2+1])+2]=[255,0,255]
        for i in range(0,temp_p):        
            img2[int(result_shape[4,i*2]):int(result_shape[4,i*2])+2,int(result_shape[4,i*2+1]):int(result_shape[4,i*2+1])+2]=[0,0,255] 
            img3[int(result_shape5[0,i*2])-3:int(result_shape5[0,i*2])+4,int(result_shape5[0,i*2+1])-3:int(result_shape5[0,i*2+1])+4]=[255,255,0]    
        #cv2.imwrite(str(index)+".jpg",new_img)
        if (compare[index, 4]!=0 and index>=0):
            img3[compare[index, 0]-1:compare[index, 0]+2, compare[index, 1]-1:compare[index, 1]+2, :]=[0, 0, 255]
            img3[compare[index, 2]-1:compare[index, 2]+2, compare[index, 3]-1:compare[index, 3]+2, :]=[0, 0, 255]
        if (index<0):
        	print "ok" 	
        	cv2.imshow("test_ok",img3)
        	cv2.waitKey(0)
        if (index>=0):
        	cv2.namedWindow("final",cv2.cv.CV_WINDOW_NORMAL)
        	cv2.imshow("wefwef",img2)
        	cv2.imshow("final",img3)
        	cv2.waitKey(30)
        point2x,point2y=new_shape[0,6*2],new_shape[0,6*2+1]
        size2x,size2y=new_shape[0,12*2]-new_shape[0,6*2],new_shape[0,12*2+1]-new_shape[0,6*2+1]
        new_img2=img[point2x:point2x+3*size2x,point2y-3:point2y+size2y*1.6-3,:]
        
        """cv2.imshow("ef",new_img)
        cv2.waitKey(0)"""
        """px2,py2=detect_eye(new_img2,sizex,int(size2x/2),new_shape[0,9*2+1]-point1y,
        new_shape[0,10*2+1]-point1y,index,show)
        px,py=detect_eye(new_img,sizex,int(sizex/2),new_shape[0,9*2+1]-point1y,
        new_shape[0,10*2+1]-point1y,index,show)"""
        if (show):
            b=np.loadtxt("face_detect/build/eye_point/BioID_"+str(index).rjust(4,'0') +".eye")
            cv2.imshow("img2",img2)
            p0x=np.round(px+point1x)
            p0y=np.round(py+point1y)
            p2x=np.round(px2+point2x)
            p2y=np.round(py2+point2y)
            d=np.sqrt((b[2]-p0y)*(b[2]-p0y)+(b[3]-p0x)*(b[3]-p0x))
            d_between=np.sqrt((b[2]-b[0])*(b[2]-b[0])+(b[3]-b[1])*(b[3]-b[1]))
            if (d/d_between>0.055):
                print b[2],p0y,b[3],p0x
                print d,d_between,index
                print d/d_between,index
                img[b[3]-1:b[3]+1,b[2]-1:b[2]+1,:]=[0,0,255]
                #img[p2x:p2x+1,p2y:p2y+1,:]=[255,0,0]
                img[b[1]-1:b[1]+1,b[0]-1:b[0]+1,:]=[0,0,255]
                #img[np.round(px+point1x):np.round(px+point1x)+1,np.round(py+point1y):np.round(py+point1y)+1,:]=[255,0,0]
                cv2.imshow("img",img)
        #return np.round(px+point1x),np.round(py+point1y),np.round(px2+point2x),np.round(py2+point2y)
        #return ddy+point1y,ddx+point1x,0,0
        #cv2.imshow("imshow3",img2)
        #cv2.imshow("imshow2",new_img)
        return 1
    else :
        return 0
def gogo():
    
    
    for i in range(begin,1521):
        print i
        #points[i,1],points[i,2],points[i,3],points[i,4]=go(i,0)
        points[i,1]=go(i,0)
        if (points[i,1]!=0):
            pupil[i-begin,4]=i  
            points[i,0]=i 
        #cv2.waitKey(0)  
        np.savetxt("pupil_by_shape.txt",pupil,delimiter=' ',fmt='%u')
        
    np.save("face_detect/points2.npy",points)
    print "save"
#gogo()
