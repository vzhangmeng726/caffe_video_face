import caffe

caffe.set_device(2)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('Vface_AE_solver.prototxt')

solver.solve()