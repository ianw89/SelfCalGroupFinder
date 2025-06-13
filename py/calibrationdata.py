import numpy as np
import sys

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from dataloc import *


def read_wp_file(fname):
    data = np.loadtxt(fname, skiprows=0, dtype='float')
    wp = data[:,1]
    wp_err = data[:,2]
    radius = data[:,0]
    return wp,wp_err,radius

class CalibrationData:
    """
    Class to specify the calibration data for the SelfCalGroupFinder.
    """
    def __init__(self, paramsfolder, magbins: np.ndarray, color_separation: np.ndarray, magcut, frac_area):
        if frac_area <= 0.0:
            raise ValueError("frac_area must be greater than 0.0")
        if len(color_separation) != len(magbins) - 1:
            raise ValueError("color_separation must have one less element than magbins")
        
        self.paramsfolder = paramsfolder
        self.rpbinsfile = os.path.join(self.paramsfolder, 'wp_rbins.dat')
        self.magbins = magbins # absolute magnitude bin definitions (edges)
        self.color_separation = color_separation # boolean array indicating whether for each L bin the wp should be red/blue seperate (True), or all together (False)
        self.magcut = magcut # apparent magnitude cut to use when calculating the volumes
        self.frac_area = frac_area # fraction of the area of the sky that is covered by the survey, multiplies into the volume
        self.zmaxes = np.array([get_max_observable_z(m, self.magcut).value for m in self.magbins[:-1]])
        self.volumes = np.array([get_volume_at_z(z, frac_area) for z in self.zmaxes])
        self.bincount = len(self.magbins) - 1
    
    def write_volume_bins_file(self, path: str):
        data = np.column_stack((self.magbins[:-1], self.zmaxes, self.volumes, self.color_separation.astype(int)))
        np.savetxt(path, data, fmt='%f %f %f %d')


    def get_wp_red(self, mag: int):
        mag = abs(mag)
        fname = os.path.join(self.paramsfolder, f'wp_red_M{mag:d}.dat')
        return read_wp_file(fname)
    
    def get_wp_blue(self, mag: int):
        mag = abs(mag)
        fname = os.path.join(self.paramsfolder, f'wp_blue_M{mag:d}.dat')
        return read_wp_file(fname)
    
    def get_wp_all(self, mag: int):
        mag = abs(mag)
        fname = os.path.join(self.paramsfolder, f'wp_all_M{mag:d}.dat')
        return read_wp_file(fname)
    
    def mag_to_idx(self, mag: float):
        m = abs(mag)
        return np.asarray(self.magbins == -m).nonzero()[0][0]

    @staticmethod
    def SDSS_4bin(magcut: float, frac_area: float):
        return CalibrationData(PARAMS_SDSS_FOLDER, np.array([-18, -19, -20, -21, -22]), np.array([True, True, True, True]),  magcut, frac_area)
    
    @staticmethod
    def SDSS_5bin(magcut: float, frac_area: float):
        return CalibrationData(PARAMS_SDSS_FOLDER, np.array([-17, -18, -19, -20, -21, -22]), np.array([True, True, True, True, True]), magcut, frac_area)

    @staticmethod
    def BGS_Y1_6bin(magcut: float, frac_area: float):
        return CalibrationData(PARAMS_BGSY1_FOLDER, np.array([-17, -18, -19, -20, -21, -22, -23]), np.array([True, True, True, True, True, True]), magcut, frac_area)

    @staticmethod
    def BGS_Y1_8bin(magcut: float, frac_area: float):
        return CalibrationData(PARAMS_BGSY1_FOLDER, np.array([-15, -16, -17, -18, -19, -20, -21, -22, -23]), np.array([False, False, True, True, True, True, True, True]), magcut, frac_area)

    @staticmethod
    def BGS_Y1mini(magcut: float, frac_area: float):
        return CalibrationData(PARAMS_BGSY1MINI_FOLDER, np.array([-17, -18, -19, -20, -21, -22, -23]), np.array([False, False, True, True, True, False]), magcut, frac_area)

    def __str__(self):
        return f"CalibrationData(paramsfolder={self.paramsfolder}, binsfile={self.rpbinsfile}, magbins={self.magbins}, magcut={self.magcut}, frac_area={self.frac_area})"