def conv2d(x, kernel, strides=(1, 1), padding='valid',
		   data_format=None, dilation_rate=(1, 1)):
	"""2D convolution.
	# Arguments
		kernel: kernel tensor.
		strides: strides tuple.
		padding: string, "same" or "valid".
		data_format: "channels_last" or "channels_first".
			Whether to use Theano or TensorFlow data format
		in inputs/kernels/outputs.
	"""
	data_format = normalize_data_format(data_format)

	image_shape = _preprocess_conv2d_image_shape(int_shape(x), data_format)
	kernel_shape = int_shape(kernel)
	if kernel_shape is None:
		kernel_shape = kernel.eval().shape  # in case of a shared variable
	kernel_shape = _preprocess_conv2d_filter_shape(kernel_shape, data_format)

	x = _preprocess_conv2d_input(x, data_format)
	kernel = _preprocess_conv2d_kernel(kernel, data_format)
	th_padding = _preprocess_padding(padding)

	conv_out = T.nnet.conv2d(x, kernel,
							 border_mode=th_padding,
							 subsample=strides,
							 input_shape=image_shape,
							 filter_shape=kernel_shape,
							 filter_dilation=dilation_rate)

	conv_out = _postprocess_conv2d_output(conv_out, x, padding,
										  kernel_shape, strides, data_format)
	return conv_out
