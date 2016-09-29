import caffe

caffe.set_device(2)
caffe.set_mode_gpu()

caffe_root = '/home/wangzd/caffe'
weights = caffe_root+'/'+'JLmodels/models/facerec_iter_162000.caffemodel'

solver = caffe.SGDSolver('Vface_AE_solver.prototxt')
solver.net.copy_from(weights)
solver.solve()