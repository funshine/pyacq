# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import struct
import numpy as np

from .streamhelpers import DataSender, DataReceiver, register_transfermode
from .arraytools import make_dtype

class RawBytesDataSender(DataSender):
    """Helper class to send data serialized over socket.
    
    Note: this class is usually not instantiated directly; use
    ``OutputStream.configure(transfermode='rawbytes')``.
    
    """
    def send(self, index, data):
        # optional pre-processing before send
        if isinstance(data, np.ndarray):
            for f in self.funcs:
                index, data = f(index, data)

        copy = self.params.get('copy', False)
        self.socket.send_multipart([index.to_bytes(8, byteorder='big'), data], copy=copy)

class RawBytesDataReceiver(DataReceiver):
    """Helper class to receive data serialized over socket.
    
    See RawBytesDataSender.
    """
    def __init__(self, socket, params):
        DataReceiver.__init__(self, socket, params)
    
    def recv(self, return_data=True):
        # receive and unpack structure
        buf, data = self.socket.recv_multipart()
        index = int.from_bytes(buf, byteorder='big')
        dtype = make_dtype(self.params['dtype']) # this avoid some bugs but is not efficient because this is call every sends...
        # dtype = self.params['dtype']

        if not return_data:
            return index, None

        return index, data


register_transfermode('rawbytes', RawBytesDataSender, RawBytesDataReceiver)
