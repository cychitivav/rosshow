import numpy as np

import rosshow.termgraphics as termgraphics
from rosshow.viewers.generic.GenericImageViewer import GenericImageViewer

def _convolve3x3(channel, kernel):
    """3x3 convolution with edge padding, done with plain numpy (no scipy dependency)."""
    padded = np.pad(channel, 1, mode = 'edge')
    out = np.zeros_like(channel)
    for i in range(3):
        for j in range(3):
            if kernel[i, j] == 0:
                continue
            out += kernel[i, j] * padded[i:i + channel.shape[0], j:j + channel.shape[1]]
    return out

def _debayer_rggb8(data, height, width):
    """
    Bilinear demosaicing of a bayer_rggb8 raw image into an H x W x 3 RGB image.
    RGGB layout:
        R G R G ...
        G B G B ...
    """
    raw = np.frombuffer(data, np.uint8).reshape((height, width)).astype(np.float32)

    r = np.zeros_like(raw)
    g = np.zeros_like(raw)
    b = np.zeros_like(raw)
    r[0::2, 0::2] = raw[0::2, 0::2]
    g[0::2, 1::2] = raw[0::2, 1::2]
    g[1::2, 0::2] = raw[1::2, 0::2]
    b[1::2, 1::2] = raw[1::2, 1::2]

    kernel_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0
    kernel_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4.0

    r = _convolve3x3(r, kernel_rb)
    g = _convolve3x3(g, kernel_g)
    b = _convolve3x3(b, kernel_rb)

    return np.clip(np.stack((r, g, b), axis = -1), 0, 255)

class ImageViewer(GenericImageViewer):
    def __init__(self, canvas, title = ""):
        def msg_decoder(msg):
            """
            Decodes a sensor_msgs/Image ROS message into a numpy H x W x 3 RGB image.
            This basically reproduces what cv_bridge does (except to RGB instead of BGR),
            but cv_bridge doesn't support python3 :-/
            """

            if msg.encoding == 'bgr8':
                image = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, 3))[:, :, ::-1]
            elif msg.encoding == 'bgra8':
                image = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, 4))[:, :, ::-1][:,:,1:]
            elif msg.encoding == 'rgb8':
                image = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'mono8' or msg.encoding == '8UC1':
                image = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width))
                image = np.stack((image,) * 3, axis = -1) # greyscale to RGB
            elif msg.encoding == 'bayer_rggb8':
                image = _debayer_rggb8(msg.data, msg.height, msg.width)
            elif msg.encoding == 'mono16' or msg.encoding == '16UC1':
                image = np.frombuffer(msg.data, np.uint16).reshape((msg.height, msg.width)).astype(np.float)
                image_max = np.percentile(image, 95)
                image_min = np.percentile(image, 5)
                image = 255*((image - image_min)/(image_max - image_min))
                image = np.clip(image, 0, 255)
                image = np.stack((image,) * 3, axis = -1) # greyscale to RGB
            else:
                print("Image encoding " + msg.encoding + " not supported yet.")
                return None

            return image

        GenericImageViewer.__init__(self, canvas, msg_decoder = msg_decoder, title = title)

