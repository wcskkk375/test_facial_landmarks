import numpy as np
import cv2
import sys
sys.path.append ('/home/wcs/opencv_assignments/face_detect')
from initialize_face import detect_face

p=28
total_img=583
def test_points():
    import cv2
    face_points=np.load("face_point.npy")
    img=cv2.imread("face_detect/build/BioID-FaceDatabase-V1.2/BioID_1108.pgm")
    #gray=cv2.cvtColor(img,cv2.cv.CV_RGB2GRAY)
    for i in range(0,20):
        y,x=face_points[1,i,0],face_points[1,i,1]
        img[x:x+5,y:y+5]=[255,0,0]
    cv2.namedWindow("imshow")
    cv2.imshow("imshow",img)
    cv2.waitKey(0)
def calculate_weights(shape):
    distance=np.zeros((1520,p,p),dtype=np.float32)
    for i in range(0,1520):
        for j in range(0,p):
            for k in range(0,p):
                distance[i,j,k]=np.abs(shape[i,j*2]-shape[i,k*2])+np.abs(shape[i,j*2+1]-shape[i,k*2+1])
    variance=np.zeros((p,p),dtype=np.float32)
    for i in range(0,p):
        for j in range(0,p):
            variance[i,j]=np.std(distance[:,i,j])
    weights=np.zeros((p),dtype=np.float32)
    for i in range(0,p):
        weights[i]=1.000/np.sum(variance[i,:])
    print weights,"wwwwww"
    np.save("weights.npy",weights)
    return weights
def vectorize_shape():
    face_points=np.load("face_point.npy")
    shape=np.zeros((1520,p*2),dtype=np.float32)
    for i in range(0,1520):
        for j in range(0,p):
            shape[i,j*2+1],shape[i,j*2]=face_points[i,j,0],face_points[i,j,1]
        #shape[i,4*2+1],shape[i,4*2]=face_points[i,14,0],face_points[i,14,1]
    weights=calculate_weights(shape)
    #weights=np.load("weights.npy")
    np.save("shape(1520,40).npy",shape)
    shape_m=np.mean(shape,axis=0)
    X0=np.sum(weights*shape[0,0::2])
    Y0=np.sum(weights*shape[0,1::2])

    affine_tran=np.zeros((1520,4))
    W=np.sum(weights)
    Z=np.sum(weights*(shape[1,0::2]*shape[1,0::2]+shape[1,1::2]*shape[1,1::2]))
    
    #print Z
    #print sum([weights[i]*shape[1,i*2]*shape[1,i*2]+weights[i]*shape[1,i*2+1]*shape[1,i*2+1] for i in range(0,5)])
    for i in range(0,1520):
        img=cv2.imread("face_detect/build/BioID-FaceDatabase-V1.2/BioID_0000.pgm")
        img2=cv2.imread("face_detect/build/BioID-FaceDatabase-V1.2/BioID_"+str(i).rjust(4,'0')+".pgm")
        new_shape=np.zeros((1,p*2),dtype=np.float32)
        X=np.sum(weights*shape[i,0::2])
        Y=np.sum(weights*shape[i,1::2])
        Z=np.sum(weights*(shape[i,0::2]*shape[i,0::2]+shape[i,1::2]*shape[i,1::2]))
        C1=np.sum(weights*(shape[0,0::2]*shape[i,0::2]+shape[0,1::2]*shape[i,1::2]))
        C2=np.sum(weights*(shape[0,1::2]*shape[i,0::2]-shape[0,0::2]*shape[i,1::2]))
        affine=np.array([[X,-Y,W,0],[Y,X,0,W],[Z,0,X,Y],[0,Z,-Y,X]])
        h=np.array([X0,Y0,C1,C2])
        
        #transform=np.dot(np.linalg.inv(affine),h)[:,0]
        transform=np.linalg.solve(affine, h)
        #print transform
        #print transform.shape
        for j in range(0,p):            
            new_shape[0,j*2]=int(transform[0]*shape[i,j*2]-transform[1]*shape[i,j*2+1]+transform[2])
            new_shape[0,j*2+1]=int(transform[1]*shape[i,j*2]+transform[0]*shape[i,j*2+1]+transform[3]) 
            y,x=shape_m[j*2+1],shape_m[j*2]
            img[x:x+2,y:y+2,:]=[255,0,0]           
            #print new_shape[0,j*2+1]
        
        shape[i,:]=new_shape[0,:]
        affine_tran[i,:]=transform[:]
    np.save("affine_tran.npy",affine_tran)
    #print new_shape
    return shape
def similarity_transform(shape10,shape20):
    shape1=np.zeros((shape10.shape[0]))
    shape1[:]=shape10[:]
    
    shape2=np.zeros((shape20.shape[0]))
    shape2[:]=shape20[:]
    center_x1=np.mean(shape1[::2])
    center_y1=np.mean(shape1[1::2])
    center_x2=np.mean(shape2[::2])
    center_y2=np.mean(shape2[1::2])
    shape1[::2]=shape1[::2]-center_x1
    shape1[1::2]=shape1[1::2]-center_y1
    shape2[::2]=shape2[::2]-center_x2
    shape2[1::2]=shape2[1::2]-center_y2
    temp1=np.zeros((2,shape10.shape[0]/2))
    temp2=np.zeros((2,shape20.shape[0]/2))
    
    temp1[0,:]=shape1[::2]
    temp1[1,:]=shape1[1::2]
    temp2[0,:]=shape2[::2]
    temp2[1,:]=shape2[1::2]
    covar1,m1=cv2.calcCovarMatrix(temp1.T,cv2.cv.CV_COVAR_COLS)    
    covar2,m2=cv2.calcCovarMatrix(temp2.T,cv2.cv.CV_COVAR_COLS)
    #covar1=np.cov(temp1,rowvar=1)
    #covar2=np.cov(temp2,rowvar=1)
    s1=np.sqrt(np.linalg.norm(covar1))
    s2=np.sqrt(np.linalg.norm(covar2))
    #s11=np.sqrt(np.linalg.norm(haha1))
    #s22=np.sqrt(np.linalg.norm(haha2))
    scale = s1 / s2 
    """ 
    aaa= np.sum(np.abs(shape1-shape2 )) 
    bbb= np.sum(np.abs(scale*shape2-shape1 )) 
    if (aaa>bbb):
        print "right"
    else: print "wrong"
    """
    shape1=shape1/s1
    shape2=shape2/s2
    num=np.sum(shape1[::2]*shape2[1::2]-shape1[1::2]*shape2[::2])
    den=np.sum(shape1[::2]*shape2[::2]+shape1[1::2]*shape2[1::2])
    norm=np.sqrt(num*num+den*den)
    sin_theta = num/norm
    cos_theta = den/norm
    tran=np.array([[scale*cos_theta,scale*sin_theta],[-scale*sin_theta,scale*cos_theta]])
    return tran
def project_shape(shape0,bounding_box,flag):
    shape=np.zeros((shape0.shape[0]))
    shape[:]=shape0[:]
    if (flag==1):
        shape[::2]=(shape[::2]-bounding_box[2])/bounding_box[3]
        shape[1::2]=(shape[1::2]-bounding_box[6])/bounding_box[7]
        return shape
    if (flag==-1):
        shape[::2]=(shape[::2])*bounding_box[3]+bounding_box[2]
        shape[1::2]=(shape[1::2])*bounding_box[7]+bounding_box[6]
        return shape    
def normalize_shape():
    delta_cy,delta_cx,delta_sx,delta_sy=0,0,0,0
    face_points=np.load("ferns/face_point.npy")
    shape=np.zeros((total_img,p*2),dtype=np.float32)
    bounding_box=np.zeros((total_img,8),dtype=np.int16)
    for i in range(0,total_img):
        for j in range(0,p):
            shape[i,j*2+1],shape[i,j*2]=face_points[i,j,0],face_points[i,j,1]
    np.save("ferns/shape(1520,40).npy",shape)
    for i in range(0,total_img):
        
        
        detect_ok=0
        filename="/home/wcs/opencv_assignments/lfw/images_reindex/"+str(i)+".jpg"
        img=cv2.imread(filename)
        
        if (img!=None):
            img0=img.copy()
            bounding_box0=detect_face(img,1)
        if (bounding_box0[0,0]!=0):
            for j in range(0,bounding_box0.shape[0]):
                  mean_x0,mean_y0,sx0,sy0=bounding_box0[j,0],bounding_box0[j,1],bounding_box0[j,2]/2,bounding_box0[j,3]/2
                  print (shape[i,40]-mean_y0-sy0*0.2)/sy0,(shape[i,41]-mean_x0)/sx0
                  if (abs(shape[i,40]-mean_y0-sy0*0.2)<sy0*0.2 and abs(shape[i,41]-mean_x0)<sx0*0.2):
                        mean_x,mean_y,sx,sy=bounding_box0[j,0],bounding_box0[j,1],bounding_box0[j,2]/2,bounding_box0[j,3]/2
                        detect_ok=1
                        
        if (detect_ok==0):
                  bounding_box[i,1]=min(shape[i,::2]) 
                  bounding_box[i,0]=max(shape[i,::2])
                  bounding_box[i,2]=(bounding_box[i,0]+bounding_box[i,1])/2*(1+(np.random.random_sample((1,5))[0,0]/10)-0.05)+delta_cy/i
                  bounding_box[i,3]=(bounding_box[i,0]-bounding_box[i,1])/2*(1+(np.random.random_sample((1,5))[0,0]/10)-0.05)+delta_sy/i
        
        
                  bounding_box[i,4]=max(shape[i,1::2])
                  bounding_box[i,5]=min(shape[i,1::2]) 
        
                  bounding_box[i,6]=(bounding_box[i,4]+bounding_box[i,5])/2*(1+(np.random.random_sample((1,5))[0,0]/10)-0.05)+delta_cx/i
                  bounding_box[i,7]=(bounding_box[i,4]-bounding_box[i,5])/2*(1+(np.random.random_sample((1,5))[0,0]/10)-0.05)+delta_sx/i
                  
        if (detect_ok==1):
            bounding_box[i,0]=mean_y+sy
            bounding_box[i,1]=mean_y-sy
            bounding_box[i,2]=mean_y
            bounding_box[i,3]=sy
            bounding_box[i,4]=mean_x+sx
            bounding_box[i,5]=mean_x-sx
            bounding_box[i,6]=mean_x
            bounding_box[i,7]=sx
        print bounding_box[i,3],bounding_box[i,7],i
        #if ( bounding_box[i,3]==np.NaN or bounding_box[i,7] ==np.NaN):
            #print i
        
        print bounding_box[i,2]
        delta_cy=bounding_box[i,2]-(max(shape[i,::2])+min(shape[i,::2]))/2+delta_cy
        delta_cx=bounding_box[i,6]-(max(shape[i,1::2])+min(shape[i,1::2]))/2+delta_cx
        delta_sy=bounding_box[i,3]-(max(shape[i,::2])-min(shape[i,::2]))/2 +delta_sy
        delta_sx=bounding_box[i,7]-(max(shape[i,1::2])-min(shape[i,1::2]))/2+delta_sx
        shape[i,::2]=(shape[i,::2]-bounding_box[i,2])/bounding_box[i,3]
        shape[i,1::2]=(shape[i,1::2]-bounding_box[i,6])/bounding_box[i,7]
        """
        if (detect_ok==0):
            cv2.rectangle(img, (bounding_box[i,6]- bounding_box[i,7],bounding_box[i,2]- bounding_box[i,3]), (bounding_box[i,6]+bounding_box[i,7],bounding_box[i,2]+bounding_box[i,3]), (0, 255, 0), 2)
            cv2.imshow("fews",img)
            cv2.waitKey(0)
        """
    mean_shape=np.mean(shape,axis=0)
    print mean_shape
    np.save("ferns/bound_box(1520,8).npy",bounding_box)
    
    return shape  
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
    new_shape[0,::2]=shape0[0,::2]-delta_x+0.2*sizex
    new_shape[0,1::2]=shape0[0,1::2]-delta_y
    #new_shape[0,28*2]=new_shape[0,28*2]+4
    #for i in range(18,22):
        #new_shape[0,i*2]=new_shape[0,i*2]-2
    return new_shape
if __name__ == '__main__':
    
    #shape=vectorize_shape()
    shape=normalize_shape()
    print shape
    np.save("ferns/align_shape(1520,40).npy",shape)
    #print shape[5,:]
    img=cv2.imread("face_detect/build/BioID-FaceDatabase-V1.2/BioID_0001.pgm")
    mean_y,mean_x,sx,sy=detect_face(img)
    shape=get_shape(np.mean(shape,axis=0),sx/2,sy/2,mean_y,mean_x)
    for i in range(0,p):
        y,x=shape[0,i*2+1],shape[0,i*2]
        img[x:x+5,y:y+5]=[255,0,0]
    cv2.namedWindow("imshow")
    cv2.imshow("imshow",img)
    cv2.waitKey(0)
