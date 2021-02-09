from helper import readData, create_frequency_map
from mne import io
import mne
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


DATA_PATH = "../Data/"
PATAINT_NUM = 1
PATAINT_FILE = "P{0:0=2d}-raw.fif".format(PATAINT_NUM)

ITER_FREQS = [
    ('Theta', 4, 7),
    ('Alpha', 8, 12),
    ('Beta', 13, 25)
]

def create_image(frequency_map,W = 100,H = 100):
    '''

    :param frequency_map: list of objects like ((band, fmin, fmax), epochs):
    :return: rgb matrix
    '''
    enhance_posision_factor = 100

    #: we split the data to the picture rgb
    reds = frequency_map[0][1]
    blue = frequency_map[1][1]
    green = frequency_map[2][1]

    sensors_positions = mne.channels.make_eeg_layout(frequency_map[0][1].info).pos
    sensors_positions *= enhance_posision_factor
    sensors_positions = sensors_positions.astype(int)

    images = []
    for epcoh_reds,epcoh_blue,epcoh_green in zip(reds,blue,green):
        #: create image on
        image = np.zeros((W, H, 3))
        #: create mean value for each epoch in freq on time
        epcoh_reds_mean_over_time = epcoh_reds.mean(axis = 1)
        epcoh_blue_mean_over_time = epcoh_blue.mean(axis = 1)
        epcoh_green_mean_over_time = epcoh_green.mean(axis = 1)

        for index in range(len(epcoh_reds_mean_over_time)):
            sensor_pos = sensors_positions[index]
            x = sensor_pos[0]
            y = sensor_pos[1]
            w = sensor_pos[2]
            h = sensor_pos[3]


            image[x:x+w,y:y+h,0] += epcoh_reds_mean_over_time[index]
            image[x:x+w,y:y+h,1] += epcoh_blue_mean_over_time[index]
            image[x:x+w,y:y+h,2] += epcoh_green_mean_over_time[index]

        image = np.abs(image)

        #normalized
        v_min = image.min(axis=(0, 1), keepdims=True)
        v_max = image.max(axis=(0, 1), keepdims=True)
        image = (image - v_min) / (v_max - v_min)

        images.append(image)
    return images



def main():
    file_name = DATA_PATH + PATAINT_FILE
    raw_data = readData(file_name,PATAINT_NUM)

    events = mne.find_events(raw_data, stim_channel='STI 014')
    frequency_map = create_frequency_map(ITER_FREQS,file_name,events)

    images = create_image(frequency_map)

    plt.imshow(images[0], interpolation='nearest')
    plt.show()

    return images

if __name__ == '__main__':
    images = main()