clear all;
addpath(['.' filesep 'Util' filesep]);
addpath(['.' filesep 'Util/jpegtbx_1.4' filesep]);
% all_algorithm_name = {'CAGI','ELA','NOI4','NOI2','ADQ1','ADQ2','BLK','CFA1','DCT','NADQ','ADQ3','CFA2','NOI1'};
all_algorithm_name = {'ADQ1'};
% all_datasetName = {'docimg_split811_test', 'Alinew_trainsplit811_test'};
all_datasetName = {'docimg_all'};
% all_spliceDataPath = {'C:/Users/Administrator/Desktop/difnet_1664x1664test_results/test_images/','C:/Users/Administrator/Desktop/difnet_1664x1664test_results/Alinew_train_split811/test_imgs/'};
all_spliceDataPath = {'E:/12/docimg/'};
for i = 1:length(all_algorithm_name)
    for j = 1:length(all_datasetName)
        Options.AlgorithmName=all_algorithm_name{i};
        Options.DatasetName = all_datasetName{j};
        Options.SplicedPath = all_spliceDataPath{j};
        Options.OutputPath = strcat('E:/11/',Options.AlgorithmName,'/',Options.DatasetName,'/');
        Options.ValidExtensions={'*.jpg','*.jpeg','*.tiff','*.tif','*.png','*.bmp','*.gif'};
        disp(strcat(Options.AlgorithmName,'----------',Options.DatasetName));
        ExtractMaps(Options);
    end
end
% 
% %The name of the algorithm. Must be the name of a subdirectory in %"Algorithms"
% Options.AlgorithmName='NOI2';
% %The name of the dataset. Only used for naming the output folders, does not
% %have to correspond to an existing path.
% % Options.DatasetName='Columb';
% Options.DatasetName = 'OnlyBorder';
% % Make sure all paths end with path separator! ("/" or "\" depending on your system)
% % Root path of the spliced images (no problem if they are further split into subfolders): 
% Options.SplicedPath='/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/testTamperJPEG/';
% % Root path of the authentic images (no problem if they are further split into subfolders):
% Options.AuthenticPath='/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/testTamperJPEG/';
% % Masks exist only for spliced images. They can be either a) placed in a
% % folder structure identical to the spliced images or b) have one single
% % png image in the current folder root to serve as a mask for the entire
% % dataset. See README for details.
% Options.MasksPath='/data1/zhuangpeiyu/BOOKCOVER_P_S_Artifical/2kindsHumanPS/onlyTamperBorder/maskNameSameTamper/';
% % Subdirectories per dataset and algorithm are created automatically, so
% % "OutputPath" should better be the root path for all outputs
% Options.OutputPath='/data1/zhuangpeiyu/data/OtherAlgorithmComplete/otherSpliceAlgorithms/NOI2/';
% % Certain algorithms (those depending on jpeg_read, like ADQ2, ADQ3 and
% % NADQ) only operate on .jpg and .jpeg files.
% Options.ValidExtensions={'*.jpg','*.jpeg','*.tiff','*.tif','*.png','*.bmp','*.gif'}; %{'*.jpg','*.jpeg'};
% 
% %Run the algorithm for each image in the dataset and save the results
% ExtractMaps(Options);
%Estimate the output map statistics for each image, and gather them in one
%list, then estimate the TP-FP curves
% Curves=CollectMapStatistics(Options);
% 
% %%%%%% Compact results to a visualizable output
% PresentationCurves.Means=CompactCurve(Curves.MedianPositives,Curves.MeanThreshValues);
% PresentationCurves.Medians=CompactCurve(Curves.MedianPositives,Curves.MedianThreshValues);
% PresentationCurves.KS=CompactCurve(Curves.KSPositives,0:1/(size(Curves.KSPositives,2)-1):1);
% 
% figure(1);
% plot(PresentationCurves.KS(2,:),PresentationCurves.KS(3,:));
% axis([0 0.5 0 1]);
% xlabel('False Positives');
% ylabel('True Positives');
% title(['KS Statistic:' Options.AlgorithmName ' ' Options.DatasetName]);
% 
% Values05=PresentationCurves.KS(3,PresentationCurves.KS(2,:)>=0.05);
% TP_at_05=Values05(end);
% disp(['True Positives at 5% False Positives: ' num2str(TP_at_05*100) '%']);
% 
% rmpath(['.' filesep 'Util/jpegtbx_1.4' filesep]);
% rmpath(['.' filesep 'Util' filesep]);

