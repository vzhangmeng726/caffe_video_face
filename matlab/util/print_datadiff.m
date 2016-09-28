function print_datadiff(net)

blobnames = net.blob_names();
for i=1:length(blobnames)
   blobname = char(blobnames(i));
   blob = net.blobs(blobname);
   
   
   blobdata = blob.get_data();
   fprintf('Blob: %s data: min:%d, max:%d, mean:%d\n',blobname,min(blobdata(:)),max(blobdata(:)),mean(blobdata(:)));
   blobdiff = blob.get_diff();
   fprintf('Blob: %s diff: min:%d, max:%d, mean:%d\n',blobname,min(blobdiff(:)),max(blobdiff(:)),mean(blobdiff(:)));
   
end

end