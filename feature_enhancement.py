


# reference from pytorch
# def squeeze_and_excitation(feature_maps, target_maps_dim):
# 	# squeeze and excitation
# 	feature_maps_squeezed = torch.sum(feature_maps.view(feature_maps.size(0), feature_maps.size(1), -1), dim=2) 
# 	feature_maps_indices = torch.topk(feature_maps_squeezed, k=target_maps_dim, largest=True, sorted=False).indices
# 	excitation_map = torch.zeros([feature_maps.size(0), target_maps_dim, feature_maps.size(2), feature_maps.size(3)])

# 	for counter_length in range(len(feature_maps_indices)):
# 		curr_map = feature_maps[counter_length]
# 		curr_indices = feature_maps_indices[counter_length]
# 		excitation_map[counter_length] = torch.index_select(curr_map, dim=0, index=curr_indices) 

# 	#print(excitation_map)
# 	#print(excitation_map.shape)
# 	excitation_map = excitation_map.cuda()
# 	return excitation_map