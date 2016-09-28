function print_param(net)

layernames = net.layer_names();
for i=1:length(layernames)
   layername = char(layernames(i));
   layer = net.layers(layername);
   layerparams = layer.params();
   for j =1:length(layerparams)
       paramdata = layerparams(j).get_data();
       fprintf('%s param %d data: min:%d, max:%d, mean:%d\n',layername,j,min(paramdata(:)),max(paramdata(:)),mean(paramdata(:)));
       paramdiff = layerparams(j).get_diff();
       fprintf('%s param %d diff: min:%d, max:%d, mean:%d\n',layername,j,min(paramdiff(:)),max(paramdiff(:)),mean(paramdiff(:)));
   end
end

end