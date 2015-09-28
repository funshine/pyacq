from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.util.mutex import Mutex

import numpy as np
import weakref
import time

from ..core import (WidgetNode, Node, register_node_type,  InputStream, OutputStream,
        ThreadPollInput, StreamConverter, StreamSplitter)

from .qoscilloscope import MyViewBox

try:
    import scipy.signal
    import scipy.fftpack
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


default_params = [
        {'name': 'xsize', 'type': 'float', 'value': 10., 'step': 0.1, 'limits' : (.1, 60)},
        {'name': 'nb_column', 'type': 'int', 'value': 1},
        {'name': 'background_color', 'type': 'color', 'value': 'k' },
        {'name': 'colormap', 'type': 'list', 'value': 'jet', 'values' : ['jet', 'gray', 'bone', 'cool', 'hot', ] },
        {'name': 'refresh_interval', 'type': 'int', 'value': 500 , 'limits':[5, 1000]},
        #~ {'name': 'display_labels', 'type': 'bool', 'value': False }, #TODO when title
        {'name' : 'timefreq' , 'type' : 'group', 'children' : [
                        {'name': 'f_start', 'type': 'float', 'value': 3., 'step': 1.},
                        {'name': 'f_stop', 'type': 'float', 'value': 90., 'step': 1.},
                        {'name': 'deltafreq', 'type': 'float', 'value': 3., 'step': 1.,  'limits' : [0.1, 1.e6]},
                        {'name': 'f0', 'type': 'float', 'value': 2.5, 'step': 0.1},
                        {'name': 'normalisation', 'type': 'float', 'value': 0., 'step': 0.1},]}
    ]

default_by_channel_params = [ 
                {'name': 'visible', 'type': 'bool', 'value': True},
                {'name': 'clim', 'type': 'float', 'value': 1.},
            ]


class QTimeFreq(WidgetNode):
    """
    Class to visulaize time-frequency morlet scalogram for multiple signal.
    
    The QTimeFreq need as inputstream:
        * transfermode==sharedarray
        * timeaxis==1
    
    If the inputstream has not this propertis  the class create it own proxy input
    with a node StreamConverter.
    
    """
    
    _input_specs = {'signal' : dict(streamtype = 'signals', shape = (-1), )}
    
    _default_params = default_params
    _default_by_channel_params = default_by_channel_params
    
    def __init__(self, **kargs):
        WidgetNode.__init__(self, **kargs)
        
        self.mainlayout = QtGui.QHBoxLayout()
        self.setLayout(self.mainlayout)
        
        self.grid = QtGui.QGridLayout()
        self.mainlayout.addLayout(self.grid)
        
        self.graphicsviews = []
    
    def show_params_controler(self):
        self.params_controler.show()
        #TODO deal with modality
    
    def _configure(self, with_user_dialog = True, max_xsize = 60., nodegroup_friends = None):
        self.with_user_dialog = with_user_dialog
        self.max_xsize = max_xsize
        self.nodegroup_friends = nodegroup_friends
    
    def _initialize(self, ):
        self.sampling_rate = sr = self.input.params['sampling_rate']
        d0, d1 = self.input.params['shape']
        if self.input.params['timeaxis']==0:
            self.nb_channel  = d1
            sharedarray_shape = (int(sr*self.max_xsize), 1)
        else:
            self.nb_channel  = d0
            sharedarray_shape = (1, int(sr*self.max_xsize)),
        print('sharedarray_shape', sharedarray_shape)
        
        #create splitter and worker
        if self.nodegroup_friends is None:
            self.splitter = StreamSplitter()
        else:
            ng = self.nodegroup_friends[0]
            self.splitter = ng.create_node('StreamSplitter')
            
        self.splitter.configure()
        self.splitter.input.connect(self.input.params)
        
        self.workers = []
        self.input_proxys = []
        self.pollers = []
        for i in range(self.nb_channel):
            
            
            if self.nodegroup_friends is None:
                self.splitter.outputs[str(i)].configure(transfermode='sharedarray', ring_buffer_method='double',
                                sharedarray_shape=sharedarray_shape, )
                worker = TimeFreqCompute(workernum = i)
            else:
                print(self.splitter.outputs)
                # FIXME : problem de outputstream inputstream proxy car after_input_connect n'est pas appel�
                self.splitter.outputs[str(i)].configure(transfermode='plaindata', protocol = 'ipc') #TODO ipc not in windows
                ng = self.nodegroup_friends[i%(len(self.nodegroup_friends)-1)+1]
                worker = ng.create_node('TimeFreqCompute', workernum = i)
                
            worker.configure(max_xsize = self.max_xsize)
            worker.input.connect(self.splitter.outputs[str(i)])
            worker.output.configure(protocol = 'ipc', transfermode = 'plaindata')#the shape is after when TimeFreqCompute.on_fly_change_wavelet()
            worker.initialize()
            self.workers.append(worker)
            
            proxy_input = InputStream()
            proxy_input.connect(worker.output)
            self.input_proxys.append(proxy_input)
            
            poller = ThreadPollInput(input_stream = proxy_input)
            poller.new_data.connect(self._on_new_map)
            poller.i = i
            self.pollers.append(poller)
        
        self.splitter.initialize()

        self.mutex_action = Mutex()
        self.actions = []
        self.timer_action = QtCore.QTimer(singleShot=True, interval = 100)
        self.timer_action.timeout.connect(self.apply_actions)
        
        # Create parameters
        all = [ ]
        for i in range(self.nb_channel):
            name = 'Signal{}'.format(i)
            all.append({ 'name': name, 'type' : 'group', 'children' : self._default_by_channel_params})
        self.by_channel_params = pg.parametertree.Parameter.create(name='AnalogSignals', type='group', children=all)
        self.params = pg.parametertree.Parameter.create( name='Global options',
                                                    type='group', children =self._default_params)
        self.all_params = pg.parametertree.Parameter.create(name = 'all param',
                                    type = 'group', children = [self.params,self.by_channel_params  ])
        self.params.param('xsize').setLimits([16./sr, self.max_xsize*.95]) 
        self.all_params.sigTreeStateChanged.connect(self.on_param_change)
        
        if self.with_user_dialog:
            self.params_controler = TimeFreqControler(parent = self, viewer = self)
            self.params_controler.setWindowFlags(QtCore.Qt.Window)
        else:
            self.params_controler = None
        
        #TODO maybe delayed this ??? (with timer)
        self.create_grid()
        self.initialize_time_freq()
        self.initialize_plots()
        
    def _start(self):
        self.splitter.start()
        for i in range(self.nb_channel):
            if self.by_channel_params.children()[i]['visible']:
                self.workers[i].start()
                self.pollers[i].start()
    
    def _stop(self):
        for i in range(self.nb_channel):
            if self.by_channel_params.children()[i]['visible']:
                self.pollers[i].stop()
                self.pollers[i].wait()
                self.workers[i].start()
        self.splitter.stop()
    
    def _close(self):
        print('ici 1')
        if self.running():
            self.stop()
        print('ici 2')
        if self.with_user_dialog:
            self.params_controler.close()
        print('ici 3')
        for worker in self.workers:
            worker.close()
        print('ici 4')
        for poller in self.pollers:
            poller.close()
        self.splitter.close()

    def create_grid(self):
        color = self.params['background_color']
        for graphicsview in self.graphicsviews:
            if graphicsview is not None:
                graphicsview.hide()
                self.grid.removeWidget(graphicsview)
        self.plots =  [ None ] * self.nb_channel
        self.graphicsviews =  [ None ] * self.nb_channel
        self.images = [ None ] * self.nb_channel
        r,c = 0,0
        for i in range(self.nb_channel):
            if not self.by_channel_params.children()[i]['visible']: continue

            viewBox = MyViewBox()
            if self.with_user_dialog:
                viewBox.doubleclicked.connect(self.show_params_controler)
            viewBox.gain_zoom.connect(self.clim_zoom)
            viewBox.xsize_zoom.connect(self.xsize_zoom)
            
            graphicsview  = pg.GraphicsView()
            graphicsview.setBackground(color)
            plot = pg.PlotItem(viewBox = viewBox)
            #~ plot.setTitle(self.stream['channel_names'][i])#TODO
            graphicsview.setCentralItem(plot)
            self.graphicsviews[i] = graphicsview
            
            self.plots[i] = plot
            self.grid.addWidget(graphicsview, r,c)
            
            c+=1
            if c==self.params['nb_column']:
                c=0
                r+=1
    
    def initialize_time_freq(self):
        tfr_params = self.params.param('timefreq')
        
        # we take sampling_rate = f_stop*4 or (original sampling_rate)
        if tfr_params['f_stop']*4 < self.sampling_rate:
            sub_sampling_rate = tfr_params['f_stop']*4
        else:
            sub_sampling_rate = self.sampling_rate
        
        # this try to find the best size to get a timefreq of 2**N by changing
        # the sub_sampling_rate and the sig_chunk_size
        self.wanted_size = self.params['xsize']
        self.len_wavelet = l = int(2**np.ceil(np.log(self.wanted_size*sub_sampling_rate)/np.log(2)))
        self.sig_chunk_size = self.wanted_size*self.sampling_rate
        self.downsampling_factor = int(np.ceil(self.sig_chunk_size/l))
        self.sig_chunk_size = self.downsampling_factor*l
        self.sub_sampling_rate  = self.sampling_rate/self.downsampling_factor
        self.plot_length =  int(self.wanted_size*sub_sampling_rate)
        
        self.wavelet_fourrier = generate_wavelet_fourier(self.len_wavelet, tfr_params['f_start'], tfr_params['f_stop'],
                            tfr_params['deltafreq'], self.sub_sampling_rate, tfr_params['f0'], tfr_params['normalisation'])
        
        if self.downsampling_factor>1:
            self.filter_b = scipy.signal.firwin(9, 1. / self.downsampling_factor, window='hamming')
            self.filter_a = np.array([1.])
        else:
            self.filter_b = None
            self.filter_a = None
        
        for worker in self.workers:
            worker.on_fly_change_wavelet(wavelet_fourrier=self.wavelet_fourrier, downsampling_factor=self.downsampling_factor,
                        sig_chunk_size = self.sig_chunk_size, plot_length=self.plot_length, filter_a=self.filter_a, filter_b=self.filter_b)
        
        for input_proxy in self.input_proxys:
            input_proxy.params['shape'] = (self.plot_length, self.wavelet_fourrier.shape[1])
            input_proxy.params['sampling_rate'] = sub_sampling_rate
            
    
    def initialize_plots(self):
        #TODO get cmap from somewhere esle
        from matplotlib.cm import get_cmap
        from matplotlib.colors import ColorConverter
        
        lut = [ ]
        cmap = get_cmap(self.params['colormap'] , 3000)
        for i in range(3000):
            r,g,b =  ColorConverter().to_rgb(cmap(i) )
            lut.append([r*255,g*255,b*255])
        self.lut = np.array(lut, dtype = np.uint8)
        
        tfr_params = self.params.param('timefreq')
        for i in range(self.nb_channel):
            if self.by_channel_params.children()[i]['visible']:
                for item in self.plots[i].items:
                    #remove old images
                    self.plots[i].removeItem(item)
                
                clim = self.by_channel_params.children()[i]['clim']
                f_start, f_stop = tfr_params['f_start'], tfr_params['f_stop']
                
                image = pg.ImageItem()
                image.setImage(np.zeros((self.plot_length,self.wavelet_fourrier.shape[1])), lut = self.lut, levels = [0,clim])
                self.plots[i].addItem(image)
                image.setRect(QtCore.QRectF(-self.wanted_size, f_start,self.wanted_size, f_stop-f_start))
                self.plots[i].setXRange( -self.wanted_size, 0.)
                self.plots[i].setYRange(f_start, f_stop)
                self.images[i] =image
                
                if self.running() and not self.workers[i].running():
                    
                    self.workers[i].start()
                    self.pollers[i].start()
                else:
                    print('not running yet')
            else:
                if self.workers[i].running():
                    self.workers[i].stop()
                    self.pollers[i].stop()
                    self.pollers[i].wait()
    
    def on_param_change(self, params, changes):
        do_create_grid = False
        do_initialize_time_freq = False
        do_initialize_plots = False
        
        for param, change, data in changes:
            if change != 'value': continue
            #immediate action
            if param.name()=='background_color':
                color = data
                for graphicsview in self.graphicsviews:
                    if graphicsview is not None:
                        graphicsview.setBackground(color)
            if param.name()=='refresh_interval':
                for worker in self.workers:
                    worker.set_interval(data)
            if param.name()=='clim':
                i = self.by_channel_params.children().index(param.parent())
                clim = param.value()
                if self.images[i] is not None:
                    self.images[i].setImage(self.images[i].image, lut = self.lut, levels = [0,clim])
            
            #difered action
            if param.name()=='xsize':
                do_initialize_time_freq = True
            if param.name()=='colormap':
                do_initialize_plots = True
            if param.name()=='nb_column':
                do_create_grid = True
            if param.name() in ('f_start', 'f_stop', 'deltafreq', 'f0', 'normalisation'):
                do_initialize_time_freq = True
            if param.name()=='visible':
                do_create_grid = True
        
        do_initialize_plots = do_initialize_plots or do_initialize_time_freq or do_create_grid
        with self.mutex_action:
            if do_create_grid:
                if self.create_grid not in self.actions:
                    self.actions.append(self.create_grid)
            if do_initialize_time_freq:
                if self.initialize_time_freq not in self.actions:
                    self.actions.append(self.initialize_time_freq)
            if do_initialize_plots:
                if self.initialize_plots not in self.actions:
                    self.actions.append(self.initialize_plots)
            if not self.timer_action.isActive():
                self.timer_action.start()
    
    def apply_actions(self):
        with self.mutex_action:
            for action in self.actions:
                action()
            self.actions = []

    def _on_new_map(self, pos, data):
        i = self.sender().i
        print('_on_new_map', 'data', id(data))
        if self.images[i] is None: return
        self.images[i].updateImage(data)

    def clim_zoom(self, factor):
        for i, p in enumerate(self.by_channel_params.children()):
            p.param('clim').setValue(p.param('clim').value()*factor)
        
    def xsize_zoom(self, xmove):
        factor = xmove/100.
        newsize = self.params['xsize']*(factor+1.)
        limits = self.params.param('xsize').opts['limits']
        if newsize>limits[0] and newsize<limits[1]:
            self.params['xsize'] = newsize    

    def auto_clim(self, identic = True):
        if identic:
            all = [ ]
            for i, p in enumerate(self.by_channel_params.children()):
                if p.param('visible').value():
                    all.append(np.max(self.images[i].image))
            clim = np.max(all)*1.1
            for i, p in enumerate(self.by_channel_params.children()):
                if p.param('visible').value():
                    p.param('clim').setValue(float(clim))
        else:
            for i, p in enumerate(self.by_channel_params.children()):
                if p.param('visible').value():
                    clim = np.max(self.images[i].image)*1.1
                    p.param('clim').setValue(float(clim))

register_node_type(QTimeFreq)


def generate_wavelet_fourier(len_wavelet, f_start, f_stop, deltafreq, sampling_rate, f0, normalisation):
    """
    Compute the wavelet coefficients at all scales and makes its Fourier transform.
    
    Parameters
    ----------
    len_wavelet : int
        length in sample of the windows
    f_start: float
        First frequency in Hz
    f_stop: float
        Last frequency in Hz
    deltafreq:
        Frequency interval in Hz
    sampling_rate:
        Sampling rate in Hz
    
    Returns:
    ----------
        wf : Fourier transform of the wavelet coefficients (after weighting)
              axis 0 is time,  axis 1 is freq
    """
    # compute final map scales
    scales = f0/np.arange(f_start,f_stop,deltafreq)*sampling_rate
    
    # compute wavelet coeffs at all scales
    xi=np.arange(-len_wavelet/2.,len_wavelet/2.)
    xsd = xi[:,np.newaxis] / scales
    wavelet_coefs=np.exp(complex(1j)*2.*np.pi*f0*xsd)*np.exp(-np.power(xsd,2)/2.)

    weighting_function = lambda x: x**(-(1.0+normalisation))
    wavelet_coefs = wavelet_coefs*weighting_function(scales[np.newaxis,:])

    # Transform the wavelet into the Fourier domain
    wf=scipy.fftpack.fft(wavelet_coefs,axis=0)
    wf=wf.conj()
    
    return wf



class ThreadCompute(QtCore.QThread):
    def __init__(self, in_stream, out_stream, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.in_stream = weakref.ref(in_stream)
        self.out_stream = weakref.ref(out_stream)
        
        self.sampling_rate = sr = self.in_stream().params['sampling_rate']
        self.interval = 500
        self.lock = Mutex()
        self.running = False
        
        self.wavelet_fourrier = None
        
    def run(self):
        import time
        with self.lock:
            self.running = True
        
        self.head = 0
        self._n = 0
        while True:
            with self.lock:
                if not self.running:
                    break
            
            while self.in_stream().poll(timeout = 0)>0:
                #~ print('recv')
                self.head, _ = self.in_stream().recv()
            #~ print('head', self.head)
            
            self.compute()
            print('sleep', self.interval/1000.)
            time.sleep(self.interval/1000.)
            

    def stop(self):
        with self.lock:
            self.running = False

    #~ def on_fly_change_wavelet(self, wavelet_fourrier=None, downsampling_factor=None, sig_chunk_size = None,
            #~ plot_length=None, filter_a=None, filter_b=None):
    def on_fly_change_wavelet(self, kargs):
        
        self.wavelet_fourrier = kargs['wavelet_fourrier']
        self.downsampling_factor = kargs['downsampling_factor']
        self.sig_chunk_size = kargs['sig_chunk_size']
        self.plot_length = kargs['plot_length']
        self.filter_a = kargs['filter_a']
        self.filter_b = kargs['filter_b']
        self.out_stream().params['shape'] = (self.plot_length, self.wavelet_fourrier.shape[1])
        self.out_stream().params['sampling_rate'] = self.sampling_rate/self.downsampling_factor
    
    def set_interval(self, interval):
        self.interval = interval
        #~ self.timer.setInterval(interval)
    
    def compute(self):
        #~ print('compute', self.wavelet_fourrier)
        if self.wavelet_fourrier is None: 
            # not on_fly_change_wavelet yet
            return
        #~ print(self.wavelet_fourrier.shape)
        
        t1 = time.time()
        head = self.head
        if self.downsampling_factor>1:
            head = head - head%self.downsampling_factor
        full_arr = self.in_stream().get_array_slice(head, self.sig_chunk_size).flatten()
        
        t2 = time.time()
        
        if self.downsampling_factor>1:
            small_arr = scipy.signal.filtfilt(self.filter_b, self.filter_a, full_arr)
            small_arr =small_arr[::self.downsampling_factor].copy()# to ensure continuity
        else:
            small_arr = full_arr
        small_arr_f=scipy.fftpack.fft(small_arr)
        #~ print('yep', self.wavelet_fourrier.shape, small_arr_f.shape)
        if small_arr_f.shape[0] != self.wavelet_fourrier.shape[0]: return
        wt_tmp=scipy.fftpack.ifft(small_arr_f[:,np.newaxis]*self.wavelet_fourrier,axis=0)
        wt = scipy.fftpack.fftshift(wt_tmp,axes=[0])
        wt = np.abs(wt).astype(self.out_stream().params['dtype'])
        wt = wt[-self.plot_length:]
        
        #send map
        self._n += 1
        self.out_stream().send(self._n, wt)
        t3 = time.time()
        print('compute', self.workernum,  t2-t1, t3-t2, t3-t1)





class TimeFreqCompute(Node):
    """
    TimeFreqCompute compute a wavelet scalogram every X interval and
    send it to QTimeFreq.
    """
    _input_specs = {'signal' : dict(streamtype = 'signals', shape = (-1), )}
    _output_specs = {'timefreq' : dict(streamtype = 'image', dtype = 'float32')}
    
    def __init__(self, workernum = None, **kargs):
        Node.__init__(self, **kargs)
        assert HAVE_SCIPY, "TimeFreqCompute node depends on the `scipy` package, but it could not be imported."
        self.workernum = workernum
    
    def _configure(self, max_xsize = 60.):
        self.max_xsize = max_xsize
    
    def after_input_connect(self, inputname):
        self.sampling_rate = sr = self.input.params['sampling_rate']
        
        assert len(self.input.params['shape']) == 2, 'Wrong shape: TimeFreqCompute'
        d0, d1 = self.input.params['shape']
        if self.input.params['timeaxis']==0:
            self.nb_channel  = d1
            self.sharedarray_shape = (int(sr*self.max_xsize), self.nb_channel)
        else:
            self.nb_channel  = d0
            self.sharedarray_shape = (self.nb_channel, int(sr*self.max_xsize)),
        assert self.nb_channel == 1, 'Wrong nb_channel: TimeFreqCompute work for one channel only'
        
        stream_spec = {}
        #~ stream_spec.update(self.input.params)
        stream_spec['port'] = '*'
        self.outputs['timefreq'] = OutputStream(spec = stream_spec)
        
        
    def _initialize(self):
        sr = self.sampling_rate
        
        
        #create proxy input
        if self.input.params['transfermode'] == 'sharedarray':
            self.proxy_input = self.input
            self.conv = None
        else:
            # if input is not transfermode creat a proxy
            self.conv = StreamConverter()
            self.conv.configure()
            self.conv.input.connect(self.input.params)
            self.conv.output.configure(protocol='inproc', interface='127.0.0.1', port='*', 
                   transfermode = 'sharedarray', streamtype = 'analogsignal',
                   dtype='float32', sampling_rate = sr,
                   shape=self.input.params['shape'], timeaxis=self.input.params['timeaxis'],
                   compression='', scale=None, offset=None, units='',
                   sharedarray_shape=self.sharedarray_shape, ring_buffer_method='double',
                   )
            self.conv.initialize()
            self.proxy_input = InputStream()
            self.proxy_input.connect(self.conv.output)
        
        self.thread = ThreadCompute(self.proxy_input, self.output)
        self.thread.workernum = self.workernum
        self.sig_on_fly_change_wavelet.connect(self.thread.on_fly_change_wavelet)
        self.sig_set_interval.connect(self.thread.set_interval)

    def _start(self):
        if self.conv is not None:
            self.conv.start()
        self.thread.start()
    
    def _stop(self):
        if self.conv is not None:
            self.conv.stop()
        self.thread.stop()
        self.thread.wait()
    
    def _close(self):
        if self.running():
            self.stop()
    
    sig_on_fly_change_wavelet = QtCore.pyqtSignal(object)
    def on_fly_change_wavelet(self, **kargs):
        self.sig_on_fly_change_wavelet.emit(kargs)
    
    sig_set_interval = QtCore.pyqtSignal(int)
    def set_interval(self, interval):
        self.sig_set_interval.emit(interval)
 
register_node_type(TimeFreqCompute)


class TimeFreqControler(QtGui.QWidget):
    def __init__(self, parent = None, viewer= None):
        QtGui.QWidget.__init__(self, parent)
        
        self._viewer = weakref.ref(viewer)
        
        #layout
        self.mainlayout = QtGui.QVBoxLayout()
        self.setLayout(self.mainlayout)
        t = 'Options for {}'.format(self.viewer.name)
        self.setWindowTitle(t)
        self.mainlayout.addWidget(QtGui.QLabel('<b>'+t+'<\b>'))
        
        h = QtGui.QHBoxLayout()
        self.mainlayout.addLayout(h)

        self.tree_params = pg.parametertree.ParameterTree()
        self.tree_params.setParameters(self.viewer.params, showTop=True)
        self.tree_params.header().hide()
        h.addWidget(self.tree_params)

        self.tree_by_channel_params = pg.parametertree.ParameterTree()
        self.tree_by_channel_params.header().hide()
        h.addWidget(self.tree_by_channel_params)
        self.tree_by_channel_params.setParameters(self.viewer.by_channel_params, showTop=True)

        v = QtGui.QVBoxLayout()
        h.addLayout(v)
        
        if self.viewer.nb_channel>1:
            v.addWidget(QtGui.QLabel('<b>Select channel...</b>'))
            names = [ p.name() for p in self.viewer.by_channel_params ]
            self.qlist = QtGui.QListWidget()
            v.addWidget(self.qlist, 2)
            self.qlist.addItems(names)
            self.qlist.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
            
            for i in range(len(names)):
                self.qlist.item(i).setSelected(True)            
            v.addWidget(QtGui.QLabel('<b>and apply...<\b>'))
        
        # 
        but = QtGui.QPushButton('set visble')
        v.addWidget(but)
        but.clicked.connect(self.on_set_visible)
        
        but = QtGui.QPushButton('Automatic clim (same for all)')
        but.clicked.connect(lambda: self.auto_clim( identic = True))
        v.addWidget(but)

        but = QtGui.QPushButton('Automatic clim (independant)')
        but.clicked.connect(lambda: self.auto_clim( identic = False))
        v.addWidget(but)
        
        v.addWidget(QtGui.QLabel(self.tr('<b>Clim change (mouse wheel on graph):</b>'),self))
        h = QtGui.QHBoxLayout()
        v.addLayout(h)
        for label, factor in [ ('--', 1./10.), ('-', 1./1.3), ('+', 1.3), ('++', 10.),]:
            but = QtGui.QPushButton(label)
            but.factor = factor
            but.clicked.connect(self.clim_zoom)
            h.addWidget(but)
    
    @property
    def viewer(self):
        return self._viewer()

    @property
    def selected(self):
        selected = np.ones(self.viewer.nb_channel, dtype = bool)
        if self.viewer.nb_channel>1:
            selected[:] = False
            selected[[ind.row() for ind in self.qlist.selectedIndexes()]] = True
        return selected
    
    def on_set_visible(self):
        # apply
        visibles = self.selected
        for i,param in enumerate(self.viewer.by_channel_params.children()):
            param['visible'] = visibles[i]

    def auto_clim(self, identic = True):
        self.viewer.auto_clim(identic=identic)

    def clim_zoom(self):
        factor = self.sender().factor
        for i, p in enumerate(self.viewer.by_channel_params.children()):
            p.param('clim').setValue(p.param('clim').value()*factor)

