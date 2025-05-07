close all; clear all;
% im='demo.jpg';
% OutputMap = analyze(im);
% imagesc(OutputMap);
% all_algorithm_name = {'NADQ'};
% all_datasetName = {'OnlyBorder','Arbitrary','NIST2016_Splice'};
% all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/tamperJPEG','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/tamperJPEG','/data1/zhuangpeiyu/imageDataBase/NC2016_Test0613/NC2016_Test0613/splice_test/tamperJPEG'};
% % all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/tamperJPEG','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/tamperJPEG'};
% Output_path = '/data1/zhuangpeiyu/data/OtherAlgorithmComplete/otherSpliceAlgorithms/';

all_algorithm_name = {'NADQ'};
% % all_datasetName = {'OnlyBorder','Arbitrary','NIST2016_Splice'};
% % % all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/tamperJPEG','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/tamperJPEG','/data1/zhuangpeiyu/imageDataBase/NC2016_Test0613/NC2016_Test0613/splice_test/tamperJPEG'};
% % all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/tamperJPEG','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/tamperJPEG'};
% all_datasetName = {'PS_border','PS_arbitrary','NIST2016_manipulation'};
% all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS12','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/quality12/tamper', '/data1/zhuangpeiyu/imageDataBase/NC2016_Test0613/NC2016_Test0613/tamper/manipulation_copy/'};
% Output_path = '/data1/zhuangpeiyu/data/OtherAlgorithmComplete/otherSpliceAlgorithms/restart/';

% all_datasetName = {'PS_border8','PS_border9','PS_border10','PS_border11',
%     'PS_arbitrary8','PS_arbitrary9','PS_arbitrary10','PS_arbitrary11'};
% all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS8',
%     '/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS9',
%     '/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS10',
%     '/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS11',
%     '/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS8', 
%     '/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS9',
%     '/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS10',
%     '/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS11',
%    };
% all_datasetName = {'PS_border8','PS_border9','PS_border10','PS_border11','PS_arbitrary8','PS_arbitrary9','PS_arbitrary10','PS_arbitrary11'};
% all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS8','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS9','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS10','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS11','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS8','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS9','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS10','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS11'};
all_datasetName = {'PS_border12','PS_arbitrary12','NIST2016_manipulation'};
all_spliceDataPath = {'/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/DifferentPSquality/PS12','/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/arbitraryTamper/DifferentPSquality/PS12','/data1/zhuangpeiyu/imageDataBase/NC2016_Test0613/NC2016_Test0613/tamper/manipulation_copy'};

Output_path = '/data1/zhuangpeiyu/data/OtherAlgorithmComplete/otherSpliceAlgorithms/restart/PostProcessing';

for i = 1:length(all_algorithm_name)
    for j = 1:length(all_datasetName)
%         rmdir(strcat(Output_path,all_algorithm_name{i},'/',all_datasetName{j},'/Output_map2/'),'s');
        mkdir(strcat(Output_path,all_algorithm_name{i},'/',all_datasetName{j},'/Output_map2/'));
        images = dir(strcat(all_spliceDataPath{j},'/*.jpg'));
        parfor image_index = 1:length(images)
            image_name = images(image_index).name;
            im1 = imread(strcat(all_spliceDataPath{j},'/',image_name));
            [row,col,ch] = size(im1);
            [OutputMap] = analyze(strcat(all_spliceDataPath{j},'/',image_name));
%             [OutputMap] = analyze('demo.jpg');
%             figure
%             imshow(OutputMap);
            max_value = max(max(OutputMap));
            min_value = min(min(OutputMap));
            output_map = (OutputMap-min_value)/(max_value-min_value);
            
            output_map = uint8(output_map*255);
            final_output = imresize(output_map,[row,col]);
            map_name = strrep(image_name,'PS','MS');
            map_name = strrep(map_name,'ps','ms');
            map_name = strrep(map_name,'.jpg','.png');
            imwrite(final_output,strcat(Output_path,all_algorithm_name{i},'/',all_datasetName{j},'/Output_map2/',map_name))
        end
    end
end