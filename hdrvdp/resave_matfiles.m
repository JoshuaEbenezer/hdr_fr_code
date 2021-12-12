clear
rootdir = './features/hdrvdp3_features';
filelist = dir(fullfile(rootdir, '*.*'));  %get list of files and folders in any subfolder
filelist = filelist(~[filelist.isdir]);  %remove folders from list

for i=1:length(filelist)
    file = fullfile(filelist(i).folder,filelist(i).name);
    data = load(file);
    for j=1:9
        dis_name = char(data.featMap.distorted_names(j,:));
        outname = fullfile('./hdrvdp3_sep_features',string(dis_name(1:end-4))+'.mat'); 
        Q = mean(nonzeros(data.featMap.Qfeatures(j,:)));
        save(outname,'Q');
    end
        
end
