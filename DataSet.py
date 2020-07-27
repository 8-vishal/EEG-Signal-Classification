import numpy
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import scale
import pandas


def summed_data(data):
    return data[0] + data[1] + data[2] + data[3] + data[4] + data[5] + data[6] + data[7] + data[8] + data[9]


def create_epochs(t_series, epoch_length):
    list_a = []
    for j in range(0, len(t_series), epoch_length):
        epoch = t_series[j:j + epoch_length]
        list_a.append(epoch)
    return list_a


def create_epoch_partedEvents(epo_series):
    events = []
    for j in range(len(epo_series)):
        rest = epo_series[j][0:5000]
        stimulus = epo_series[j][5000:6000]
        get_ready = epo_series[j][6000:8000]
        imagined_speech = epo_series[j][8000:13000]
        speak = epo_series[j][13000:14000]
        events.append(rest)
        events.append(stimulus)
        events.append(get_ready)
        events.append(imagined_speech)
        events.append(speak)
    return events


def butter_LowPass(series, N, Wn):
    # N = 2  # Filter order
    # Wn = 0.01  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    y = filtfilt(B, A, series)
    return y


def imagined_speechPart(epoched_parted_series):
    list_1 = []
    indexes = numpy.arange(3, len(epoched_parted_series), 5)
    for i in indexes:
        list_1.append(epoched_parted_series[i])
    return list_1


def final_data(img_series):
    val = imagined_speechPart(img_series)
    val = val[0:145]
    val = numpy.array(val)
    return val


def time_series():
    person1 = create_epoch_partedEvents(
        create_epochs(butter_LowPass(summed_data(numpy.load("Data/Person1.npy")), 2, 0.01), 14000))
    person2 = create_epoch_partedEvents(
        create_epochs(butter_LowPass(summed_data(numpy.load("Data/Person2.npy")), 2, 0.01), 14000))
    person3 = create_epoch_partedEvents(
        create_epochs(butter_LowPass(summed_data(numpy.load("Data/Person3.npy")), 2, 0.01), 14000))
    person4 = create_epoch_partedEvents(
        create_epochs(butter_LowPass(summed_data(numpy.load("Data/Person4.npy")), 2, 0.01), 14000))
    person5 = create_epoch_partedEvents(
        create_epochs(butter_LowPass(summed_data(numpy.load("Data/Person5.npy")), 2, 0.01), 14000))
    person6 = create_epoch_partedEvents(
        create_epochs(butter_LowPass(summed_data(numpy.load("Data/Person6.npy")), 2, 0.01), 14000))
    person7 = create_epoch_partedEvents(
        create_epochs(butter_LowPass(summed_data(numpy.load("Data/Person7.npy")), 2, 0.01), 14000))
    person8 = create_epoch_partedEvents(
        create_epochs(butter_LowPass(summed_data(numpy.load("Data/Person8.npy")), 2, 0.01), 14000))
    return person1, person2, person3, person4, person5, person6, person7, person8


def class_seperator(final, start, end, step):
    b = []
    a = numpy.arange(start, end, step)
    for i in a:
        b.append(final[i])
    return numpy.array(b)


def index_list(length):
    a = []
    for i in range(93):
        a.append('iy')
    for i in range(93):
        a.append('uw')
    for i in range(93):
        a.append('piy')
    for i in range(length):
        a.append('tiy')
    for i in range(length):
        a.append('diy')
    for i in range(length):
        a.append('m')
    for i in range(length):
        a.append('n')
    for i in range(length):
        a.append('pat')
    for i in range(length):
        a.append('pot')
    for i in range(length):
        a.append('knew')
    for i in range(length):
        a.append('gnaw')
    return a

def matrx_selector(array):
    array = numpy.array(array)
    array = array[0:28]
    return array


def extract_data(filename):
    labels = []
    features = []
    for line in pandas.read_csv(filename):
        row = line.split(',')
        labels.append(row[0])
        features.append(x for x in row[1:len(row)])
    return labels, features



a, b, c, d, e, f, g, h = time_series()

a1 = final_data(a)
b1 = final_data(b)
c1 = final_data(c)
d1 = final_data(d)
e1 = final_data(e)
f1 = final_data(f)
g1 = final_data(g)
h1 = final_data(h)

y = numpy.vstack((a1, b1, c1, d1, e1, f1, g1))
iy = class_seperator(y, 0, 1015, 11)
#iy = iy[0:84]
print(iy.shape)
uw = class_seperator(y, 1, 1015, 11)
#uw = uw[0:84]
print(uw.shape)
piy = class_seperator(y, 2, 1015, 11)
#piy = piy[0:84]
print(piy.shape)
tiy = class_seperator(y, 3, 1015, 11)
#tiy = tiy[0:84]
print(tiy.shape)
diy = class_seperator(y, 4, 1015, 11)
#diy = diy[0:84]
print(diy.shape)
m = class_seperator(y, 5, 1015, 11)
#m = m[0:84]
print(m.shape)
n = class_seperator(y, 6, 1015, 11)
#n = n[0:84]
print(n.shape)
pat = class_seperator(y, 7, 1015, 11)
#pat = pat[0:84]
print(pat.shape)
pot = class_seperator(y, 8, 1015, 11)
#pot = pot[0:84]
print(pot.shape)
knew = class_seperator(y, 9, 1015, 11)
#knew = knew[0:84]
print(knew.shape)
gnaw = class_seperator(y, 10, 1015, 11)
#gnaw = gnaw[0:84]
print(gnaw.shape)
fg = numpy.vstack((iy, uw, piy, tiy, diy, m, n, pat, pot, knew, gnaw))
fg = scale(fg)
index = index_list(length=92)

print(fg.shape)

df = pandas.DataFrame(fg, index=index)
ds = df.sample(frac=1)
ds.to_csv("Data/CSV/fulltrain.csv")
#x = df.iloc[0:922, 0]
#y.fit(df)
#print(y.categories_)

