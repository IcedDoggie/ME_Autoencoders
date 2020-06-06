def visualize_class_activation_maps(weights_path, model, designated_layer, img_list, img_labels, ori_img, ori_img_path):
	sys.setrecursionlimit(10000)
	no_of_subj = []
	identity_arr = []

	for root, folders, files in os.walk(weights_path):
		no_of_subj = len(files)
	# print(no_of_subj)
	# get subject and vid identity
	for root, folders, files in os.walk(ori_img_path):
		if len(root) > 85:
			# print(root)

			identity_idx = root.split('/', 9)
			identity = str(identity_idx[-2]) + '_' + str(identity_idx[-1])
			identity_arr += [identity]
			# print(identity)

	temp_identity_counter = 0
	for counter in range(no_of_subj):
		weights_name = weights_path + str(counter) + '.h5'
		model.load_weights(weights_name)

		gen = create_generator_LOSO(img_list, img_labels, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		ori_gen = create_generator_LOSO(ori_img, img_labels, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		for (alpha, beta) in zip(gen, ori_gen):
			X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			X_ori, _, _ = beta[0], beta[1], beta[2]
			print(X.shape)
			print(X_ori.shape)

			predicted_labels = np.argmax(model.predict(X), axis=1)
			non_binarized_y = non_binarized_y[0]

			# visualize CAM
			for img_counter in range(len(predicted_labels)):
				input_img = X[img_counter]
				input_img = input_img.reshape((1, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
				predict = np.argmax(model.predict(input_img), axis=1)

				# utils.apply_modifications(model)
				layer_idx = utils.find_layer_idx(model, 'activation_1')
				# layer_idx = model.layers[-1]
				penultimate_layer = utils.find_layer_idx(model, 'max_pooling2d_3')
				# penultimate_layer = model.layers[-5]
				cam = vi.visualize_cam(model, layer_idx=layer_idx, filter_indices=predict, seed_input=input_img, penultimate_layer_idx=penultimate_layer, \
					backprop_modifier=None, grad_modifier=None)

				# layer_idx = utils.find_layer_idx(model, 'conv_2')
				# cam = vi.visualize_activation(model, layer_idx=layer_idx, filter_indices=None, seed_input=None, \
				# 	backprop_modifier=None, grad_modifier=None)				

				# reshape operation
				input_img = input_img.reshape((input_img.shape[1], input_img.shape[2], input_img.shape[3]))
				input_img = np.transpose(input_img, (1, 2, 0))
				gray_img = X_ori[img_counter]
				gray_img = np.transpose(gray_img, (1, 2, 0))
				# Plotting
				fig, axes = plt.subplots(1, 6, figsize=(18, 6))
				print(input_img.shape)
				txt_X = 0
				txt_Y = 60
				# Reverse Discretization
				predict = reverse_discretization(predict[0])
				label = reverse_discretization(non_binarized_y[img_counter])
				predict_str = "Predicted: " + predict
				label_str = "Label: " + label

				identity_str = identity_arr[temp_identity_counter]
				temp_identity_counter += 1
				
				# plt.text(txt_X, txt_Y, identity_str, fontsize=15)
				# plt.text(txt_X, txt_Y * 3, predict_str, fontsize=15)
				# plt.text(txt_X, txt_Y * 4, label_str, fontsize=15)

				# axes[0].imshow(np.uint8(input_img))
				# axes[1].imshow(cam)
				# axes[2].imshow(vi.overlay(cam, input_img))
				# axes[3].imshow(np.uint8(gray_img))
				# axes[4].imshow(vi.overlay(cam, gray_img))
				# axes[5].imshow(input_img)

				plt.imshow(vi.overlay(cam, gray_img))	
				plt.tick_params(
					axis='both',          # changes apply to the x-axis
					which='both',      # both major and minor ticks are affected
					bottom=False,      # ticks along the bottom edge are off
					top=False,         # ticks along the top edge are off
					left=False,
					right=False,
					labelleft=False,
					labelbottom=False,) # labels along the bottom edge are off

				# for ax in axes:
				# 	ax.set_xticks([])
				# 	ax.set_yticks([])
				# 	ax.grid(False)

				identity_str = identity_str + "_predict_" + predict + "_label_" + label
				save_str = '/media/ice/OS/Datasets/Visualizations/CAM_AlexNet_50/' + identity_str + '.png'
				plt.savefig(save_str)
				print(str(img_counter) + ' / ' + str(len(predicted_labels)))
				# plt.show()


			print("Predicted: ")
			print(predicted_labels)
			print("GroundTruth: ")
			print(non_binarized_y)