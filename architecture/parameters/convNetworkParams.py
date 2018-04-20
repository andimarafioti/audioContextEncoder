__author__ = 'Andres'


class ConvNetworkParams(object):
    def __init__(self, filterShapes, channels, strides, name):
        self._filterShapes = filterShapes
        self._channels = channels
        self._strides = strides
        self._name = name

    def filterShapes(self):
        return self._filterShapes

    def channels(self):
        return self._channels

    def inputChannels(self):
        return self._channels[:-1]

    def outputChannels(self):
        return self._channels[1:]

    def strides(self):
        return self._strides

    def name(self):
        return self._name

    def layerCount(self):
        return len(self._strides)

    def convNames(self):
        return ["Conv_"+str(index) for index in range(self.layerCount())]
