
import numpy as np

class camera:
    def __init__(self,fx,fy,cx,cy,p1=0,p2=0,k1=0,k2=0,k3=0):
        
        self.fx=fx
        self.fy=fy
        self.cy=cx
        self.cy=cy
        self.mat=np.ndarray(shape=(3,3),dtype=float)

        self.mat[0][0]=fx
        self.mat[0][1]=0
        self.mat[0][2]=cx

        self.mat[1][0]=0
        self.mat[1][1]=fy
        self.mat[1][2]=cy

        self.mat[2][0]=0
        self.mat[2][1]=0
        self.mat[2][2]=1

        self.distortion=np.ndarray(shape=(1,5),dtype=float)
        self.distortion[0][0]=k1
        self.distortion[0][0]=k2
        self.distortion[0][0]=p1
        self.distortion[0][0]=k2
        self.distortion[0][0]=k3

    def matrix(self):
        return self.mat

    def dis_x(self):
        return self.distortion
