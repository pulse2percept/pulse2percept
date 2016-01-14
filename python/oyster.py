import numpy as np

def jansonius(ang0,r,r0 = 4,center = np.array([15,2]),rot = 0*np.pi/180,
              scale = 1,bs = -1.9, bi = .5, cropRad = 30):
    ''' Implements the model of retinal axonal pathways by generating a 
     matrix of (x,y) positions.  See:
     
     Jansonius et al., 2009, A mathematical description of nerve fiber bundle trajectories and their variability
     in the human retina, Vision Research
    '''
    
    # Default parameters:
    #
    # r0 = 4;             %Minumum radius (optic disc size)
    # 
    # center = [15,2];    %p.center of optic disc
    # 
    # rot = 0*pi/180;    %Angle of rotation (clockwise)
    # scale = 1;             %Scale factor
    # 
    # bs = -1.9;          %superior 'b' parameter constant
    # bi = .5;            %inferior 'c' parameter constant

    
    # generate angle and radius matrices from vectors with meshgrid
    ang0mat,rmat = np.meshgrid(ang0,r)
    
    nR = ang0mat.shape[0]
    nCells =ang0mat.shape[1]

    # index into superior (upper) axons
    sup = ang0mat>0  
    
    # Set up 'b' parameter:
    b =  np.zeros([nR,nCells])
    
    b[sup] = np.exp(bs+3.9*np.tanh(-(ang0mat[sup]-121)/14))  # equation 5
    b[~sup] = -np.exp(bi+1.5*np.tanh(-(-ang0mat[~sup]-90)/25)) # equation 6
    
    # Set up 'c' parameter:
    c = np.zeros([nR,nCells])
    
    c[sup] = 1.9+1.4*np.tanh((ang0mat[sup]-121)/14);  # equation 3 (fixed typo)
    c[~sup] = 1+.5*np.tanh((-ang0mat[~sup]-90)/25);   # equation 4

    # %Here's the main function: spirals as a function of r (equation 1)
    ang = ang0mat + b*(rmat-r0)**c;
    
    # Transform to x-y coordinates
    xprime = rmat*np.cos(ang*np.pi/180);                
    yprime = rmat*np.sin(ang*np.pi/180);  

    # Find where the fibers cross the horizontal meridian 
    cross = np.zeros([nR,nCells])
    cross[sup] = yprime[sup]<0
    cross[~sup] = yprime[~sup]>0
    
    # Set Nans to axon paths after crossing horizontal meridian
    id = np.where(np.transpose(cross))    
    
    currCol = -1
    for i in range(0,len(id[0])):  #loop through axons
        if currCol != id[0][i]:
            yprime[id[1][i]:,id[0][i]] = np.NaN
            currCol = id[0][i]
        
    # Bend the image according to (the inverse) of Appendix A
    xmodel = xprime+center[0]
    ymodel = yprime
    id = xprime>-center[0]
    ymodel[id] = yprime[id] + center[1]*(xmodel[id]/center[0])**2

    #  rotate about the optic disc and scale 
    x = scale*(np.cos(rot)*(xmodel-center[0])+np.sin(rot)*(ymodel-center[1])) + center[0]
    y = scale*(-np.sin(rot)*(xmodel-center[0]) + np.cos(rot)*(ymodel-center[1])) + center[1]
    
    # Crop into a 30 x 30 deg disc
    id = x**2+y**2 > cropRad**2
    x[id] = np.NaN
    y[id] = np.NaN    
    
    return x,y

def findNearestPixel(x,y,xa,ya):
    #find the nearest axon to this pixel
    d = (x-xg.flat[id])**2+ (y-yg.flat[id])**2
    return np.unravel_index(np.nanargmin(d),d.shape)
        