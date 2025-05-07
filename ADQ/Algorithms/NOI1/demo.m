% The demo images is taken from the dataset used in:
% Fontani, Marco, Tiziano Bianchi, Alessia De Rosa, Alessandro Piva, and 
% Mauro Barni. "A framework for decision fusion in image forensics based on 
% Dempster–Shafer theory of evidence." Information Forensics and Security, 
% IEEE Transactions on 8, no. 4 (2013): 593-607.
% Original image name: Forgery_final 11.jpg
% Dataset available at: http://clem.dii.unisi.it/~vipp/index.php/imagerepos
% itory/129-a-framework-for-decision-fusion-in-image-forensics-based-on-dem
% pster-shafer-theory-of-evidence
% and the Columbia Uncompressed Image Splicing Detection Evaluation Dataset
% Original image name: canonxt_kodakdcs330_sub_01
% http://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/

close all; clear all;
im='demo.jpg';
% im='demo.tif';
OutputMap = analyze(im);
imagesc(OutputMap);

all_algorithm_name = {'NOI1'};
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