import os
import sys
import numpy as np
import h5py
import csv
import PVGeo
from PVGeo.filters import VoxelizePoints

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    # www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    # zipfile = os.path.basename(www)
    # os.system('wget %s; unzip %s' % (www, zipfile))
    # os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    # os.system('rm %s' % (zipfile))


def shuffle_data(coord_pc, vel_pc, labels): # N x NUM_POINT x 3
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[0]) 
    np.random.shuffle(idx)
    return coord_pc[idx, ...], vel_pc[idx, ...], labels[idx,...], idx

def random_vel(vel_pc):
    
    method = np.random.choice(2, 1) # 0 or 1
    level = 0.025
    #clip = 0.05
    if method == 0: # Gaussian noise
        new_vel_pc = vel_pc + np.random.normal(0,level,(vel_pc.shape[0], vel_pc.shape[1], vel_pc.shape[2]))
        
    elif method ==1: # White noise
        new_vel_pc = vel_pc + level * np.random.randn(vel_pc.shape[0], vel_pc.shape[1], vel_pc.shape[2])
    
    return new_vel_pc

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def noise_point_cloud(batch_data, noise=0.01):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    jittered_data = noise * batch_data.mean() * np.random.randn(batch_data.shape[0], batch_data.shape[1], batch_data.shape[2])
    jittered_data += batch_data
    return jittered_data


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    
    aneurysmfile = f'../data/{filename}_aneurysm.csv'
    #print(aneurysmfile)
    x_star = []
    y_star = [] # N x 1
    z_star = []
    U_star = []
    V_star = [] # N x 1
    W_star = []
    x=[]
    y=[]
    z=[]
    u=[]
    v=[]
    w=[]
    # p=[]
      
        
    with open(aneurysmfile) as f:
        reader = csv.reader(f)
        i=0
        for r in reader:
          i=i+1
          #print(np.array(r).dtype)
          if ((i >=7) & ((np.array(r).dtype == '<U16') or (np.array(r).dtype == '<U17') or (np.array(r).dtype == '<U9'))): #& (len(r)>=3): #& (np.array(r).dtype == 'float')):
            # print(i)
            #print(np.array(r).dtype)
            # dummy
            x.append(float(r[0]))
            y.append(float(r[1]))
            z.append(float(r[2]))
              
            u.append(float(r[5])) # u
            v.append(float(r[6])) # v
            w.append(float(r[7])) # w
            # p.append(float(r[6]))
        
        if np.array(x).mean() > 10.0:
            print(aneurysmfile)
            x = [x_*0.001 for x_ in x]
            y = [y_*0.001 for y_ in y]
            z = [z_*0.001 for z_ in z]
        
        x_star=np.concatenate([x_star,x],0)
        y_star=np.concatenate([y_star,y],0)
        z_star=np.concatenate([z_star,z],0)
        U_star=np.concatenate([U_star,u],0)
        V_star=np.concatenate([V_star,v],0)
        W_star=np.concatenate([W_star,w],0)
        # P_star=np.concatenate([P_star,p],0)
    
    #print(len(x_star))     
    X_star=np.array(x_star) #* 0.001
    Y_star=np.array(y_star)
    Z_star=np.array(z_star)
    
    
    U_star=np.array(U_star)
    V_star=np.array(V_star)
    W_star=np.array(W_star)
    # P_star=np.array(P_star)
    
    X_star = X_star.reshape(-1,1)
    Y_star = Y_star.reshape(-1,1)
    Z_star = Z_star.reshape(-1,1)
    
    U_star = U_star.reshape(-1,1)
    V_star = V_star.reshape(-1,1)
    W_star = W_star.reshape(-1,1)
    # P_star = P_star.reshape(-1,1)
    
    # points = np.concatenate((X_star,Y_star,Z_star),axis=1)
    # pts = PVGeo.points_to_poly_data(points)
    # pts['u'] = U_star.flatten(order="F") 
    # pts['v'] = V_star.flatten(order="F") 
    # pts['w'] = W_star.flatten(order="F") 
    
    # pts.save("{}_tot.vtk".format("mesh"))
    
    # # 1024 points selected for each case randomly
    #idx_data = np.random.choice(X_star.shape[0], 1024)

    # points = np.concatenate((X_star[idx_data,:],Y_star[idx_data,:],Z_star[idx_data,:]),axis=1)
    
    # pts = PVGeo.points_to_poly_data(points)
    # pts['u'] = U_star[idx_data,:].flatten(order="F") 
    # pts['v'] = V_star[idx_data,:].flatten(order="F") 
    # pts['w'] = W_star[idx_data,:].flatten(order="F") 
    
    # pts.save("{}_1024.vtk".format("mesh"))
    
    # idx_data = np.random.choice(X_star.shape[0], 256)
    
    # points = np.concatenate((X_star[idx_data,:],Y_star[idx_data,:],Z_star[idx_data,:]),axis=1)
    
    # pts = PVGeo.points_to_poly_data(points)
    # pts['u'] = U_star[idx_data,:].flatten(order="F") 
    # pts['v'] = V_star[idx_data,:].flatten(order="F") 
    # pts['w'] = W_star[idx_data,:].flatten(order="F") 
    
    # pts.save("{}_256.vtk".format("mesh"))
    # dummy

    # current_data = np.concatenate([X_star.reshape(-1,1), Y_star.reshape(-1,1), Z_star.reshape(-1,1)],1)
    # current_label = np.concatenate([U_star.reshape(-1,1), V_star.reshape(-1,1), W_star.reshape(-1,1)],1)
    
    # current_data = np.expand_dims(current_data,axis = 0)
    # current_label = np.expand_dims(current_label,axis = 0)
    
    X_star = np.expand_dims(X_star,axis = 0)
    Y_star = np.expand_dims(Y_star,axis = 0)
    Z_star = np.expand_dims(Z_star,axis = 0) # 1 x N x 1
    
    U_star = np.expand_dims(U_star,axis = 0)
    V_star = np.expand_dims(V_star,axis = 0)
    W_star = np.expand_dims(W_star,axis = 0)
    # P_star = np.expand_dims(P_star,axis = 0)
    
    # print(X_star)
    # print(X_star.min())
    # print(X_star.max())
    
    # X_star = (X_star - X_star.min()) / (X_star.max() - X_star.min())
    # Y_star = (Y_star - Y_star.min()) / (Y_star.max() - Y_star.min())
    # Z_star = (Z_star - Z_star.min()) / (Z_star.max() - Z_star.min())
    
    # U_star = (U_star - U_star.min()) / (U_star.max() - U_star.min())
    # V_star = (V_star - V_star.min()) / (V_star.max() - V_star.min())
    # W_star = (W_star - W_star.min()) / (W_star.max() - W_star.min())
    
    #coord_pc = np.concatenate([X_star, Y_star, Z_star],2) # 1 x N x 3
    #vel_pc = np.concatenate([U_star, V_star, W_star],2)
    
    return X_star,Y_star,Z_star,U_star,V_star,W_star


    #return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
