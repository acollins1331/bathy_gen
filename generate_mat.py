#! /usr/bin/env python

from netCDF4 import Dataset, num2date, date2num
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat

filename='ensemble100.nc'

def read_ensemble_array_from_file(filename):
    dataset = Dataset(filename)
    ensemble_filename = filename
    read_array_from_file(filename, dataset)
    ensemble_array = dataset['ensemble_array'][:]
    return ensemble_array

def read_array_from_file(filename, dataset):
    nx = dataset.dimensions['x'].size
    ny = dataset.dimensions['y'].size
    nn = dataset.dimensions['nn'].size
    x = dataset['xFRF']
    y = dataset['yFRF']

def write_ensemble_array_to_nc(filename=None):
    filename = filename or 'ensemble.nc'
    if not os.path.exists(filename):
        write_array_to_nc(
            filename, ensemble_array, 'ensemble_array')
    else:
        print('File {} exists.'.format(filename))

def write_array_to_nc(filename, array, variable_name):
    ncfile = Dataset(filename, 'w')
    x_dim = ncfile.createDimension('x', self.nx)
    y_dim = ncfile.createDimension('y', self.ny)
    nn = ncfile.createDimension('nn', self.nn)
    ncol = ncfile.createDimension(
        'ncol', array.shape[1])
    
    nc_array = ncfile.createVariable(
        variable_name, 'f4', ('nn', 'ncol'))
    nc_array.units = 'meter'

    xfrf = ncfile.createVariable('xFRF', 'f4', ('x'))
    xfrf.units = 'meter'
    
    yfrf = ncfile.createVariable('yFRF', 'f4', ('y'))
    yfrf.units = 'meter'

    nc_array[:] = array
    
    xfrf[:] = np.arange(-50.0, 1305.0, 5.0)
    yfrf[:] = np.arange(-200.0, 5000.0, 5.0) 
    
    ncfile.close()

def write_array_to_mat(ensemble_array):
    example_mat = loadmat('./celeris_bathy.mat')
    #print(example_mat.keys())
    #print(example_mat['B'].shape)
    #print(example_mat['B'])
    #create new mat file by taking the example mat and overwriting just the 'B' bathy section
    #but the array is only of size 271 wtf?
    #imgplot = plt.imshow(example_mat['B'])
    #imgplot = plt.pcolormesh(example_mat['x'].squeeze(), example_mat['y'].squeeze(), example_mat['B'].T)
    #plt.show()
    print(ensemble_array.shape)
    for n in range(len(ensemble_array[0])):
        k=0
        new_array = np.zeros((361,271), dtype=np.double)
        for i in range(361):
            for j in range(271):
                new_array[i,j] = ensemble_array[k][n]
                k+=1
        new_array = new_array.T
        new_array = new_array[:194,:]
        new_mat = example_mat
        new_mat['B'] = new_array
        #print(new_mat.keys())
        #print(new_mat['B'].shape)
        #print(new_mat['B'])
        #print(example_mat['B'].shape)
        #imgplot = plt.imshow(new_array)
        #plt.show()
        savemat('celeris_gen_bathy' + str(n) + '.mat', new_mat)
        saved_mat = loadmat('./celeris_gen_bathy' + str(n) + '.mat')
        print(saved_mat.keys())
        print(saved_mat['B'].shape)
        print(saved_mat['B'])
        #imgplot = plt.imshow(saved_mat['B'])
        #plt.show()

ensemble_array = read_ensemble_array_from_file(filename)
write_array_to_mat(ensemble_array)

    
        
