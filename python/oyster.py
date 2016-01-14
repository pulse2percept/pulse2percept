import numpy as np

def jansonius(ang0,r,r0 = 4,center = np.array([15,2]),rot = 0*np.pi/180,
              scale = 1,bs = -1.9, bi = .5):
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
    
 
    
    return x,y

def makeAxonStreaks(xg,yg,xa,ya,axon_lambda=1,min_weight = .001):
 
    #initialize lists
    axon_xg = ()
    axon_yg = ()
    axon_dist = ()
    axon_weight = ()
    axon_id = ()

    #loop through pixels as indexed into a single dimension
    for id in range(0,len(xg.flat)-1):
        
        #find the nearest axon to this pixel
        d = (xa-xg.flat[id])**2+ (ya-yg.flat[id])**2        
        cur_ax_id = np.nanargmin(d) #index into the current axon   
        [axPosId0,axNum] = np.unravel_index(cur_ax_id,d.shape)  
        
        dist = 0

        cur_xg = xg.flat[id]
        cur_yg = yg.flat[id]        
        
        #add first values to the list for this pixel
        axon_dist = axon_dist + ([0],) 
        axon_weight = axon_weight + ([1],)
        axon_xg = axon_xg + ([cur_xg],)
        axon_yg = axon_yg + ([cur_yg],)
        axon_id = axon_id + ([id],)
        
        # plt.plot(xa[:axPosId0,axNum],ya[:axPosId0,axNum],'.-')    
        
        #now loop back along this nearest axon toward the optic disc         
        for axPosId in range(axPosId0-1,-1,-1):
            #increment the distance from the starting point
            dist = dist + np.sqrt((xa[axPosId+1,axNum]-xa[axPosId,axNum])**2
            + (ya[axPosId+1,axNum]-ya[axPosId,axNum])**2)
            
            weight = np.exp(-dist/axon_lambda)  # weight falls off exponentially as distance from axon cell body
    
            #find the nearest pixel to the current position along the axon
            nearest_xg_id = np.abs(xg[0,:]-xa[axPosId,axNum]).argmin()
            nearest_yg_id = np.abs(yg[:,0]-ya[axPosId,axNum]).argmin()
            nearest_xg =xg[0,nearest_xg_id]
            nearest_yg =yg[nearest_yg_id,0]                    
    
            #if the position along the axon has moved to a new pixel, and the weight isn't too small...
            if nearest_xg != cur_xg or nearest_yg != cur_yg and weight>min_weight:
                #update the current pixel location
                cur_xg = nearest_xg
                cur_yg = nearest_yg
    
                #append the list 
                #axon_dist[id].append(dist)
                axon_weight[id].append(np.exp(-dist/axon_lambda))
                #axon_xg[id].append(cur_xg)
                #axon_yg[id].append(cur_yg)
                axon_id[id].append(np.ravel_multi_index((nearest_yg_id,nearest_xg_id),xg.shape)) 
                
    return axon_id, axon_weight