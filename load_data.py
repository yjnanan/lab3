import struct
import numpy as np

#load Images
def loadImage(filename):
    binfile=open(filename,'rb')
    buffers=binfile.read()
    head=struct.unpack_from('>IIII',buffers,0)
    offset=struct.calcsize('>IIII')
    imgNum=head[1]
    width=head[2]
    height=head[3]
    #[60000]*28*28
    bits=imgNum*width*height
    bitsString='>'+str(bits)+'B'
    imgs=struct.unpack_from(bitsString,buffers,offset)
    binfile.close()
    imgs=np.reshape(imgs,[imgNum,width*height])
    imgs=imgs.astype(np.float)
    imgs=imgs/255.0
    return imgs

#load labels
def loadLabel(filename):
    binfile=open(filename,'rb')
    buffers=binfile.read()
    head=struct.unpack_from('>II',buffers,0)
    imgNum=head[1]
    offset=struct.calcsize('>II')
    numString='>'+str(imgNum)+"B"
    labels=struct.unpack_from(numString,buffers,offset)
    binfile.close()
    labels=np.reshape(labels,[imgNum])
    binlabels = np.zeros((imgNum,10))
    for i,s in enumerate(labels): binlabels[i,s]=1
    return binlabels
