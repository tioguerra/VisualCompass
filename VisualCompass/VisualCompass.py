import numpy as np
import cv2

MIN_MATCHES = 10

class VisualCompass(object):
    ''' This class implements a compass
    based on images of the ceiling for
    robotics applications. One needs to
    provide a series of pictures of the
    ceiling always with the north towards
    the top of the image. Then later, an
    arbitrary image can be provided and
    the rotation angle will be estimated
    '''
    ref_sifts = [] # list of tuples kp, des
    sift = None # SIFT detector object
    flann = None # Flann matcher object
    def __init__(self, ref_imgs):
        ''' This constructor gets a list of
        image files. The images should be .png
        format pictures of the ceiling, with
        the north towards the top.
        '''
        # Create the SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()
        # Create the point cloud matcher
        index_params = dict(algorithm = 0, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Find the SIFT points in the ref_imgs
        self._calculateSifts(ref_imgs)
    def _calculateSifts(self, imgs):
        ''' This method is called by the
        constructor. It finds the SIFT
        feature points of the reference
        images and stores them for later
        reference
        '''
        for img in imgs:
            mat = cv2.imread(img,0)
            kp, des = self.sift.detectAndCompute(mat,None)
            self.ref_sifts.append((kp, des))
    def getNorth(self, img_mat):
        ''' This method tries to estimate an
        angle for the north, given a sample
        image (img_mat, OpenCV matrix format).
        The angle is given in radians, with
        positive angles meaning clockwise
        rotation of the north (counter clock-
        wise rotation of the robot).
        '''
        kp2, des2 = self.sift.detectAndCompute(img_mat, None)
        results = []
        for kp1, des1 in self.ref_sifts:
            matches = self.flann.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            results.append((kp1, des1, good))
        kp1, des1, good = max(results, key=lambda x: x[2])
        if len(good) > MIN_MATCHES:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M = cv2.estimateRigidTransform(src_pts, dst_pts, False)
            angle = np.arctan2(M[1,0],M[0,0])
            return angle
        else:
            return None

