import numpy as np
import cv2
filename_cascade="/home/wcs/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(filename_cascade)

def detect_face(img,default=0):
    #filename="face_detect/build/BioID-FaceDatabase-V1.2/BioID_"+str(index).rjust(4,'0')+".pgm"
    #img=cv2.imread(filename)
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(70, 70))
    if (len(rects)== 0 and default==0) :
        return 0,0,0,0
    if (len(rects)== 0 and default==1):
        return np.zeros((1,4))
    rects[:,2:] += rects[:,:2]
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #cv2.imshow("imshow",img)
    #cv2.waitKey(0)
    if (default==1 ):
      bounding_box=np.zeros((len(rects),4),dtype=np.int16)
      i=0
      for x1, y1, x2, y2 in rects: 
            bounding_box[i,0],bounding_box[i,1],bounding_box[i,2],bounding_box[i,3]=np.round(x1/2+x2/2),np.round(y1/2+y2/2),x2-x1,y2-y1  
            i+=1    
      return bounding_box
    if (default==0 ):      
      return np.round(x1/2+x2/2),np.round(y1/2+y2/2),x2-x1,y2-y1
def detect_eye(img0,sizex,y0,x0,x1,index,show):
    
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img=np.asarray(img,dtype=np.float32)
    img=cv2.GaussianBlur(img,(3,3),0.5)
    
    lx=cv2.Sobel(img,-1,1,0,ksize=5)
    ly=cv2.Sobel(img,-1,0,1,ksize=5)
    lxx=cv2.Sobel(img,-1,2,0,ksize=5)
    lyy=cv2.Sobel(img,-1,0,2,ksize=5)
    lxy=cv2.Sobel(img,-1,1,1,ksize=5)
    c_factor=-(lx*lx+ly*ly)/(ly*ly*lxx-2*lx*ly*lxy+lx*lx*lyy+1)
    dx=lx*c_factor
    dy=ly*c_factor
    weight=lxx*lxx+2*lxy*lxy+lyy*lyy
    weight[:,:x0]=0
    weight[:,x1:]=0
    weight[:,:np.round(img.shape[1]/3)]=0
    weight[:,np.round(img.shape[1]/3*2):]=0
    #weight[:y0,:]=0
    weight[3.5*y0:,:]=0
    vote_map=np.zeros((img.shape[0],img.shape[1]))
    
    for i in range(0,img.shape[0]):
        for j in range(int(x0),int(x1)):
            if (np.round(dy[i,j]+i)<img.shape[0]-1 and np.round(dy[i,j]+i)>=4 and 
                np.round(dx[i,j]+j)<img.shape[1]-1 and np.round(dx[i,j]+j)>=4):
                dis=np.sqrt(dy[i,j]*dy[i,j]+dx[i,j]*dx[i,j])
                if (dis>2 and dis<8 ):
                    vote_map[np.round(dy[i,j]+i),np.round(dx[i,j]+j)]+=np.sqrt(weight[i,j])
                    """img1=img0.copy()
                    img1[i,j,:]=[255,0,0]
                    img1[np.round(dy[i,j]+i),np.round(dx[i,j]+j),:]=[0,0,255]
                    cv2.imshow("efef",img1)
                    cv2.waitKey(0)
                    if (dy[i,j]>0 and dx[i,j]>0 and img[i+1,j]-img[i,j]<0 and  img[i,j+1]-img[i,j]<0):
                        
                        
                        vote_map[np.round(dy[i,j]+i),np.round(dx[i,j]+j)]+=np.sqrt(weight[i,j])
                    if (dy[i,j]<0 and dx[i,j]>0 and img[i-1,j]-img[i,j]<0 and  img[i,j+1]-img[i,j]<0):
                        vote_map[np.round(dy[i,j]+i),np.round(dx[i,j]+j)]+=np.sqrt(weight[i,j])
                    if (dy[i,j]>0 and dx[i,j]<0 and img[i+1,j]-img[i,j]<0 and  img[i,j-1]-img[i,j]<0):
                        vote_map[np.round(dy[i,j]+i),np.round(dx[i,j]+j)]+=np.sqrt(weight[i,j])
                    if (dy[i,j]<0 and dx[i,j]<0 and img[i-1,j]-img[i,j]<0 and  img[i,j-1]-img[i,j]<0):
                        vote_map[np.round(dy[i,j]+i),np.round(dx[i,j]+j)]+=np.sqrt(weight[i,j])"""
    
    #cv2.waitKey(0)
    
    #vote_map=cv2.medianBlur(vote_map,5)
    vote_map=cv2.blur(vote_map,(6,6))
    vote_map[:,:x0]=0
    vote_map[:,x1:]=0
    vote_map[:,:np.round(img.shape[1]/3)]=0
    vote_map[:,np.round(img.shape[1]/3*2):]=0
    vote_map[:y0,:]=0
    vote_map[3.5*y0:,:]=0
    indice=np.argmax(vote_map)
    point_y=indice/img.shape[1]
    point_x=indice%img.shape[1]
    #print indice,point_y,point_x
    img0[point_y:point_y+1,point_x:point_x+1,:]=[0,255,0]
    
    weight=np.asarray(np.round(weight/np.max(weight)*255),dtype=np.uint8)
    """circles=cv2.HoughCircles(img0[:,:,0],method=cv2.cv.CV_HOUGH_GRADIENT,
    dp=8.0,minDist=1) 
    print circles
    for x,y,r in circles:
        img0[y,x,:]=[0,0,255]"""
    dx=np.asarray(dx,dtype=np.uint8)
    vote_map=np.asarray(np.round(vote_map/np.max(vote_map)*255),dtype=np.uint8)
    lxx=np.asarray(np.round(lxx/np.max(lxx)*255),dtype=np.uint8)
    #print vote_map
    if (show):
        cv2.imshow("dx",dx)
        cv2.imshow("lxx",lxx)
        cv2.imshow("vote",vote_map)
        cv2.imshow("weight",weight)
        cv2.imshow("img0",img0)
        
        
    return point_y,point_x
#if __name__ == '__main__':
    #for i in range(0,100):
        #detect_face(i)

