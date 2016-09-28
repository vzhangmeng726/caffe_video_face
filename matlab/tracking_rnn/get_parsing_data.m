function [datalist]=get_parsing_data(datafolder,listfile)


fid=fopen(listfile,'r');
traindata={};
count=1;
while feof(fid) == 0
   tline = fgetl(fid);
   S = regexp(tline, ' ', 'split');
   imname = S{1};
   labelname=S{2};
   imname = [datafolder imname];
   labelname = [datafolder labelname];
   traindata(count).im=imname;
   traindata(count).lb=labelname;
   count = count +1;
end
fclose(fid); 
datalist = traindata;

end