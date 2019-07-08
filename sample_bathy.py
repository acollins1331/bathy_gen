#! /usr/bin/env python

from netCDF4 import Dataset, num2date, date2num
import numpy as np
import matplotlib.pyplot as plt
import os

class SampleBathy:

    def read_survey_array_from_file(self, filename):
        self.dataset = Dataset(filename)
        self.survey_filename = filename
        self.read_array_from_file(filename)
        self.survey_array = self.dataset['survey_array'][:]

    def read_ensemble_array_from_file(self, filename):
        self.dataset = Dataset(filename)
        self.ensemble_filename = filename
        self.read_array_from_file(filename)
        self.ensemble_array = self.dataset['ensemble_array'][:]
        
    def read_array_from_file(self, filename):
        self.nx = self.dataset.dimensions['x'].size
        self.ny = self.dataset.dimensions['y'].size
        self.nn = self.dataset.dimensions['nn'].size
        self.x = self.dataset['xFRF']
        self.y = self.dataset['yFRF']

    def compute_survey_mean(self):
        self.survey_mean = np.mean(self.survey_array, axis=1)
        
    def compute_ensemble_mean(self):
        self.ensemble_mean = np.mean(self.ensemble_array, axis=1)

    def compute_anomaly(self):
        self.compute_survey_mean()
        mean = np.empty(self.survey_array.shape).T
        mean[:] = self.survey_mean
        mean = mean.T
        self.anomaly = self.survey_array - mean

    def generate_ensemble(self, n, stdev=1, n_svec=None):
        ensemble_array = np.empty([self.survey_array.shape[0], n])
        self.compute_svd()
        u, s, v = self.svd
        singular_vectors = u[:,:n_svec]
        scaling = s[:n_svec]*1./np.sqrt(self.survey_array.shape[1]-1.)
        for idx in range(n):
            weights = np.random.randn(n_svec)*stdev*scaling
            realization = singular_vectors.dot(weights) + self.survey_mean
            ensemble_array[:, idx] = realization
        self.ensemble_array = ensemble_array

    def compute_svd(self):
        self.compute_anomaly()
        self.svd = np.linalg.svd(self.anomaly, full_matrices=False)

    def write_ensemble_array_to_nc(self, filename=None):
        filename = filename or 'ensemble.nc'
        if not os.path.exists(filename):
            self.write_array_to_nc(
                filename, self.ensemble_array, 'ensemble_array')
        else:
            print('File {} exists.'.format(filename))
    
    def write_array_to_nc(self, filename, array, variable_name):
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
        
    def plot_scree(self):
        try:
            self.svd
        except:
            self.compute_svd()
        u, s, v = self.svd
        percent_variance_explained = s*s/s.dot(s)
        plt.scatter(range(len(s)), percent_variance_explained)
        plt.xlabel("singular value")
        plt.ylabel("% variance explained")
        plt.show()

    def plot(self, member, title=None):
        member = member.reshape(self.ny, self.nx)
        # try:
        #     levels = np.linspace(
        #         self.survey_array.min(), self.survey_array.max(), 20)  
        # except:
        #     levels = np.linspace(
        #         self.ensemble_array.min(), self.ensemble_array.max(), 20)
        levels = np.linspace(-13.0, 6.0, 20)
        extent = [self.x[0], self.x[-1], self.y[0], self.y[-1]]
        fig = plt.figure(figsize=(3,6))
        ax = fig.add_subplot(111)
        p = ax.contourf(member, levels=levels, extent=extent)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        fig.tight_layout()
        fig.colorbar(p)
        # plt.imshow(member, extent=extent, origin='lower')
        # x, y = np.meshgrid(self.x, self.y)
        # plt.contour(x, y, member, alpha=1.0)
        plt.show()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate an ensemble of bathymetry realizations.')
    parser.add_argument('--filename','-f',
                        help='netCDF file containing survey array.',
                        action='store',
                        required=False)#True)
    parser.add_argument('--n_members','-n',
                        help="Number of ensemble members to generate.",
                        action="store",
                        default=100)
    parser.add_argument('--stdev','-s',
                        help=('Standard deviation of the distribution used ' +
                              'to generate weights.'),
                        action="store",
                        default=25)
    parser.add_argument('--n_svec','-nv',
                        help=('Number of singular vectors to use from the ' +
                              'anomaly SVD.'),
                        action='store',
                        default=10)
    parser.add_argument('--output_filename', '-o',
                        help='Name of output netCDF file',
                        action='store',
                        default=None)
    
    opts = parser.parse_args()
    
    be = SampleBathy()
    # be.read_survey_array_from_file(opts.filename)
    # be.generate_ensemble(
    #     opts.n_members, stdev=opts.stdev, n_svec=opts.n_svec)
    # be.write_ensemble_array_to_nc(filename=opts.output_filename)
    be.read_ensemble_array_from_file('ensemble.nc')
    be.plot(be.ensemble_array[:,10])
    
        
    
    
        
