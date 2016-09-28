function layer_id =layer_name2id(layernames,layername)
for i = 1:length(layernames)
    if strcmp(layernames{i} , layername)
        layer_id = i;
        break;
    end
end
end