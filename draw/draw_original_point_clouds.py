import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py

# The part number to be drawn
numofpart = 1

# Location of .h5
h5file=r'demodata.h5'

# Parts to be drawn, e.g. 'Part1', 'Part2', 'Part3',etc.
partname = 'Part1'

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    partnames = f['name'][:]
    return data, partnames

numofpart = float(partname[4:])
Origindatas, partnames = load_h5(h5file)
partnameslist = partnames.tolist()

if numofpart in partnames:
    indexofpart = partnameslist.index(numofpart)
    label=[]
    x=[];y=[];z=[]

    OneOrgindata = Origindatas[indexofpart,:]

    for k in range(len(OneOrgindata)):
        thisFaceLabel = OneOrgindata[k]
        thisFaceLabellist = thisFaceLabel.tolist()
        thisFaceData = OneOrgindata[k,:]
        thisFaceDatalist = thisFaceData.tolist()
        for h in range(len(thisFaceDatalist)):
                x.append(thisFaceDatalist[h][0])
                y.append(thisFaceDatalist[h][1])
                z.append(thisFaceDatalist[h][2])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x, y ,z ,'bo',color='green',markersize=2)
    plt.title('point_cloud')
    plt.show()
else:
    print("The part is not in the .h5 file!")