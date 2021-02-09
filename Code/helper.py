import math as m
from mne import io
import mne

def readData(path, patientID):
    BAD_CH = ['P8', 'P10', 'T8']

    raw = io.read_raw_fif(path, preload=True)

    if patientID == 9:
        badCH = BAD_CH + ["EXG5"]
    else:
        badCH = BAD_CH
    raw.filter(1, 20, fir_design='firwin')
    raw.info['bads'] = badCH
    return raw

def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)

def create_frequency_map(iter_freqs,raw_fname,events,event_id = 22,
                         tmin = -0.1, tmax = 2,baseline = None):
    frequency_map = list()

    for band, fmin, fmax in iter_freqs:
        # (re)load the data to save memory
        raw = io.read_raw_fif(raw_fname)
        raw.load_data()

        # bandpass filter
        raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1)  # in each band and skip "auto" option.

        # epoch
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,preload=True)

        #remove ground chanels
        epochs.pick_types(eeg=True)

        #:TODO read more about
        # remove evoked response
        # epochs.subtract_evoked()

        # get analytic signal (envelope)
        # epochs.apply_hilbert(envelope=True)
        frequency_map.append(((band, fmin, fmax), epochs))
        del epochs
    del raw

    return frequency_map
