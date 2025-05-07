close all; clear all;
% im1='demo.jpg'
% [OutputMap] = analyze(im1);
% imagesc(OutputMap);

all_algorithm_name = {'ELA'};
all_datasetName = {'OnlyBorder','Arbitrary','NIST2016_Splice'};
all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/tamperJPEG','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/tamperJPEG','/data1/zhuangpeiyu/imageDataBase/NC2016_Test0613/NC2016_Test0613/splice_test/tamperJPEG'};
% all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/tamperJPEG','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/tamperJPEG'};
Output_path = '/data1/zhuangpeiyu/data/OtherAlgorithmComplete/otherSpliceAlgorithms/';
for i = 1:length(all_algorithm_name)
    for j = 1:length(all_datasetName)
        rmdir(strcat(Output_path,all_algorithm_name{i},'/',all_datasetName{j},'/Output_map/'),'s');
        mkdir(strcat(Output_path,all_algorithm_name{i},'/',all_datasetName{j},'/Output_map/'));
        images = dir(strcat(all_spliceDataPath{j},'/*.jpg'));
        for image_index = 1:length(images)
            image_name = images(image_index).name;
            im1 = imread(strcat(all_spliceDataPath{j},'/',image_name));
            [row,col,ch] = size(im1);
            [OutputMap] = analyze(strcat(all_spliceDataPath{j},'/',image_name));
            max_value = max(max(OutputMap));
            min_value = min(min(OutputMap));
            output_map = (OutputMap-min_value)/(max_value-min_value);
            output_map = uint8(output_map*255);
            final_output = imresize(output_map,[row,col]);
            map_name = strrep(image_name,'PS','MS');
            map_name = strrep(map_name,'ps','ms');
            map_name = strrep(map_name,'.jpg','.png');
            imwrite(final_output,strcat(Output_path,all_algorithm_name{i},'/',all_datasetName{j},'/Output_map/',map_name))
        end
    end
end