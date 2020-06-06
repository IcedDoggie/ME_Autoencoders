import argparse
from architecture_experiments_single_class_apex import train
from networks import train_res50_imagenet, train_vgg16_imagenet, train_inceptionv3_imagenet, train_alexnet_imagenet, train_shallow_alexnet_imagenet


# train(type_of_test, train_id, preprocessing_type, classes, \
# 	feature_type = 'grayscale', \
# 	db='Combined Dataset', spatial_size = 224, classifier_flag = 'svc', \
# 	tf_backend_flag = False, attention=False, freeze_flag = 'last'):

# train(train_shallow_alexnet_imagenet_with_attention, 'test_attention', \
# 	net=None, feature_type = 'flow_strain', db='Combined_Dataset_Apex_Flow', \
# 	spatial_size = 227, classifier_flag='svc', tf_backend_flag = False, \
# 	attention = False, freeze_flag = '2nd_last')

def main(args):
	# print(args[0]['train'])

	train(args.type_of_test, args.train_id, args.preprocessing_type, \
		args.classes, args.feature_type, args.db, args.spatial_size, \
		args.classifier_flag, args.tf_backend_flag, args.attention, args.freeze_flag)
	# train_smic(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--type_of_test', default='train_vgg16_imagenet', help='Using which model')
	parser.add_argument('--train_id', type=str, default="0", help='To name the weights of model')
	parser.add_argument('--preprocessing_type', type=str, default="vgg", help='preprocessing function')
	parser.add_argument('--classes', type=int, default=5, help='No. Classes')
	parser.add_argument('--feature_type', type=str, default='flow', help='Using which script to train.')
	parser.add_argument('--db', type=str, default='Combined Dataset', help='DB to use')
	parser.add_argument('--spatial_size', type=int, default=224, help='Image resolution')
	parser.add_argument('--classifier_flag', type=str, default='svc', help='SVC or Softmax')
	parser.add_argument('--tf_backend_flag', type=bool, default=False, help='Using TF backend')
	parser.add_argument('--attention', type=bool, default=False, help='Using attention or not')
	parser.add_argument('--freeze_flag', type=str, default='last', help='Freeze Flag')

	# FUNCTION_MAP = {
	# 	'type_of_test' : train_res50_imagenet(),
	# 	'train_id' : 0,
	# 	'preprocessing_type' : 'res',
	# 	'classes' : 5,
	# 	'feature_type' : 'flow',
	# 	'db' : 'Combined_Dataset_Apex_Flow',
		


	# }

	args = parser.parse_args()
	print(args)

	main(args)