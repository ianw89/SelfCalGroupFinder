import numpy as np



#Root finder that uses the secant method
#Finds the x that solve y=func(x,aux) to x= +/- xtol given initial guesses x0 and x1.
#All of y,aux,x0,x1 can be numpy arrays and it will solve all the equations simultaneously.
#It uses a mask to only iterate those functions that haven't yet converged
def root_sec(func,aux,y,xtol,x0,x1):
  xnew=np.copy(x0) # will be used to return the roots
  y0=func(x0,aux) # function at initial value x0
  y1=func(x1,aux) # function at initial value x1
  mask=np.abs(x1-x0)>xtol # mask to keep/update only those beyond tolerance
  i=0  # iteration counter
  itmax=100 #maximum number of iterations
  while ((np.abs(x0-x1).max()>xtol) & (i<itmax)): #iterate until all functions are converged
    # update to a new value xnew using the secant method
    xnew[mask]=x1[mask]-(y1[mask]-y[mask])*(x1[mask]-x0[mask])/(y1[mask]-y0[mask])
    x0[mask]=x1[mask]  
    x1[mask]=xnew[mask]
    y0[mask]=y1[mask] 
    y1[mask]=func(x1[mask],aux[mask]) #function at latest value
    mask=np.abs(x0-x1)>xtol  #don't iterate functions that are already converged
    i += 1
    print(np.count_nonzero(mask),' live after ',i,' iterations')
  if (i==itmax): print('not converged after',i,'iterations')  
  return xnew


#Root finder that uses the IPT method.
#See https://en.wikipedia.org/wiki/ITP_method
#Finds the x that solve y=func(x,aux) to x= +/-eps given initial bracketing values a and b.
#All of y,aux,a,b can be numpy arrays and it will solve all the equations simultaneously.
#Tuning parameters set according to Oliveira and Takahashi(2021)
#It is guaranteed to converge in no more than nmax steps.
def root_itp(f,aux,y,eps,a,b): 
  
    ya=f(a,aux)-y #function values at the initial interval
    yb=f(b,aux)-y #boundaries which must bracket the root
    bmask=(ya*yb>0) # check the a,b interval bounds the root
    absbma=abs(b-a)
    if(bmask.any()): 
        print('need to adjust the input a and b values in the following case(s):')
        for i in range(ya.size):
            #print("i   a     b   y(a)    y(b)")
            if (ya[i]*yb[i]>0):
                print(i,a[i],b[i],ya[i],yb[i])
        #print(np.where(bmask)[0],'y(a=',a[bmask],')=',ya[bmask],' and y(b=',b[bmask],')=',yb[bmask])
        raise SystemExit('as one or more roots are not bracketed') 
       
        
    #Three tuning parameters of the ITP root finding routine.
    k1=0.1/abs(b-a) # 0 to infinity
    k2=2.0 # 1 to 2.618
    n0=1   # 0 to infinity
    
    rmask=(ya>yb) 
    if(rmask.any()): 
        print('reversing some bracketing values so that yb>ya in all cases')
        #print('swapping',np.where(rmask)[0])
        #print('from',a[rmask],b[rmask])
        t=b[rmask]
        b[rmask]=a[rmask]
        a[rmask]=t
        t=yb[rmask]
        yb[rmask]=ya[rmask]
        ya[rmask]=t
        #print('to  ',a[rmask],b[rmask])
        
        
    xhalf=np.copy(a)  # initialize arrays to the right length the values are irrelevant
    xf=np.copy(a) 
    xt=np.copy(a)
    yitp=np.copy(a)
    xitp=np.copy(a)
    r=np.copy(a)
    d=np.copy(a)
    sig=np.copy(a)
    absbma=np.abs(b-a) #width of initial bracketed region
    nmax=np.rint(np.log10(absbma/eps)/np.log10(2.0)+0.5).astype(int)+n0 #absolute maximum number of iterations necessary
    print('root_itp: maximum number of iterations required=',nmax.max()+1)
    j=0 #iteration counter
    mask=(absbma>2.0*eps) #mask used to restrict to cases not yet converged
  

    while ( ((absbma>2.0*eps).any()) ):#loop until all cases converge
        xhalf[mask]=(a[mask]+b[mask])/2.0 #midpoint of bracketed region
        r[mask]=eps*(2**(nmax[mask]-j))-absbma[mask]/2.0 #projection range
        d[mask]=k1[mask]*(absbma[mask]**k2)       #maximum adjustment allowed relative to midpoint
        #Interpolation
        xf[mask]=(yb[mask]*a[mask]-ya[mask]*b[mask])/(yb[mask]-ya[mask])  #interpolation secant estimate 
        #Truncation
        sig[mask]=np.sign(xhalf[mask]-xf[mask])   #sign indicating which side of midpoint we have landed
        xt[mask]=xhalf[mask]           #adopt the midpoint unless
        umask=mask & (d<=abs(xhalf-xf))#this is further away than d
        xt[umask]=xf[umask]+sig[umask]*d[umask] #in which case move by just d in direction of midpoint
        #Projection 
        xitp[mask]=xt[mask]            #default new value is value xt after truncation
        umask=mask & (abs(xt-xhalf)>r) #but we only keep this if within r of midpoint
        xitp[umask]=xhalf[umask]-sig[umask]*r[umask]# otherwise we just go r from the midpoint
        #Update interval
        yitp[mask]=f(xitp[mask],aux[mask])-y[mask] #evaluate function at this new trial point 
        umask=mask & (yitp>0)                      #if greater than 0 use to update point b
        b[umask]=xitp[umask] 
        yb[umask]=yitp[umask]
        umask=mask & (yitp<0)                      #else if less than 0 use to update point a
        a[umask]=xitp[umask] 
        ya[umask]=yitp[umask]
        umask=mask & (yitp==0)  #if exactly zero update both a and b to this position as were done
        a[umask]=xitp[umask]
        b[umask]=xitp[umask]  
        absbma[mask]=np.abs(b[mask]-a[mask])  #upate the bracket region and loop back
        j += 1
        mask=(absbma>2.0*eps)  #only iterate non-converged cases
        #print('j=',j,'a:',a,'b:',b,'abs(b-a)',absbma,(absbma>2.0*eps),(absbma>2.0*eps).any())
        #
        print ('iteration',j,':',np.count_nonzero(absbma>2.0*eps),' not yet converged')
    #print('converged after',j,' iterations',a,b,ya,yb)   
    return (a+b)/2.0    


# 2nd version just parsing a second auxilary parameter
#Root finder that uses the IPT method.
#See https://en.wikipedia.org/wiki/ITP_method
#Finds the x that solve y=func(x,aux) to x= +/-eps given initial bracketing values a and b.
#All of y,aux,a,b can be numpy arrays and it will solve all the equations simultaneously.
#Tuning parameters set according to Oliveira and Takahashi(2021)
#It is guaranteed to converge in no more than nmax steps.
#
#If a root is not bracket this version gives a single warning but then returns b in place of the root!
def root_itp2(f,aux,aux2,y,eps,a,b): 
  
    ya=f(a,aux,aux2)-y #function values at the initial interval
    yb=f(b,aux,aux2)-y #boundaries which must bracket the root
    absbma=abs(b-a)
    bmask=(ya*yb>0) # check the a,b interval bounds the root
    a[bmask]=b[bmask]  #collapse the interval and effectively return b in place of the root if the root is not bracketed by [a,b]
    ya[bmask]=f(a[bmask],aux[bmask],aux2)-y[bmask] #update ya for these cases 

    if(bmask.any()): 
        print('Warning: Some roots were not bracketed by [a,b] and b is being returned in place of the root!')
#        for i in range(ya.size):
#            if (ya[i]*yb[i]>0):
#                print(i,"   a     b   y(a)    y(b)")
#                print(i,a[i],b[i],ya[i],yb[i])
#        #print(np.where(bmask)[0],'y(a=',a[bmask],')=',ya[bmask],' and y(b=',b[bmask],')=',yb[bmask])
#        raise SystemExit('as one or more roots are not bracketed') 
       
        
    #Three tuning parameters of the ITP root finding routine.
    k1=0.1/absbma # 0 to infinity
    k2=2.0 # 1 to 2.618
    n0=1   # 0 to infinity
    
    rmask=(ya>yb) 
    if(rmask.any()): 
        print('reversing some bracketing values so that yb>ya in all cases')
        #print('swapping',np.where(rmask)[0])
        #print('from',a[rmask],b[rmask])
        t=b[rmask]
        b[rmask]=a[rmask]
        a[rmask]=t
        t=yb[rmask]
        yb[rmask]=ya[rmask]
        ya[rmask]=t
        #print('to  ',a[rmask],b[rmask])
        
        
    xhalf=np.copy(a)  # initialize arrays to the right length the values are irrelevant
    xf=np.copy(a) 
    xt=np.copy(a)
    yitp=np.copy(a)
    xitp=np.copy(a)
    r=np.copy(a)
    d=np.copy(a)
    sig=np.copy(a)
    nmax=np.rint(np.log10(absbma/eps)/np.log10(2.0)+0.5).astype(int)+n0 #absolute maximum number of iterations necessary
    absbma=np.abs(b-a) #width of initial bracketed region
    print('root_itp2: maximum number of iterations required=',nmax.max()+1)
    j=0 #iteration counter
    mask=(absbma>2.0*eps) #mask used to restrict to cases not yet converged
  

    while ( ((absbma>2.0*eps).any()) ):#loop until all cases converge
        xhalf[mask]=(a[mask]+b[mask])/2.0 #midpoint of bracketed region
        
        # TEST: 29.4.24
        # nmax = float(nmax)
        
        r[mask]=eps*(2**(nmax[mask]-j))-absbma[mask]/2.0 #projection range
        
        
        
        
        d[mask]=k1[mask]*(absbma[mask]**k2)       #maximum adjustment allowed relative to midpoint
        #Interpolation
        xf[mask]=(yb[mask]*a[mask]-ya[mask]*b[mask])/(yb[mask]-ya[mask])  #interpolation secant estimate 
        #Truncation
        sig[mask]=np.sign(xhalf[mask]-xf[mask])   #sign indicating which side of midpoint we have landed
        xt[mask]=xhalf[mask]           #adopt the midpoint unless
        umask=mask & (d<=abs(xhalf-xf))#this is further away than d
        xt[umask]=xf[umask]+sig[umask]*d[umask] #in which case move by just d in direction of midpoint
        #Projection 
        xitp[mask]=xt[mask]            #default new value is value xt after truncation
        umask=mask & (abs(xt-xhalf)>r) #but we only keep this if within r of midpoint
        xitp[umask]=xhalf[umask]-sig[umask]*r[umask]# otherwise we just go r from the midpoint
        #Update interval
        yitp[mask]=f(xitp[mask],aux[mask],aux2)-y[mask] #evaluate function at this new trial point 
        umask=mask & (yitp>0)                      #if greater than 0 use to update point b
        b[umask]=xitp[umask] 
        yb[umask]=yitp[umask]
        umask=mask & (yitp<0)                      #else if less than 0 use to update point a
        a[umask]=xitp[umask] 
        ya[umask]=yitp[umask]
        umask=mask & (yitp==0)  #if exactly zero update both a and b to this position as were done
        a[umask]=xitp[umask]
        b[umask]=xitp[umask]  
        absbma[mask]=np.abs(b[mask]-a[mask])  #upate the bracket region and loop back
        j += 1
        mask=(absbma>2.0*eps)  #only iterate non-converged cases
        #print('j=',j,'a:',a,'b:',b,'abs(b-a)',absbma,(absbma>2.0*eps),(absbma>2.0*eps).any())
        #
        print ('iteration',j,':',np.count_nonzero(absbma>2.0*eps),' not yet converged')
    #print('converged after',j,' iterations',a,b,ya,yb)   
    return (a+b)/2.0    
