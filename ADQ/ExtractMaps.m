function ExtractMaps( Options )
    AlgorithmName=Options.AlgorithmName;
    DatasetName=Options.DatasetName;
    SplicedPath=Options.SplicedPath;
%     AuthenticPath=Options.AuthenticPath;
%     MasksPath=Options.MasksPath;
    
    SplicedOutputPath=[Options.OutputPath DatasetName AlgorithmName filesep 'Sp' filesep];
%     AuthenticOutputPath=[Options.OutputPath DatasetName AlgorithmName filesep 'Au' filesep];
    ValidExtensions=Options.ValidExtensions;
    
    SplicedList={};
%     AuthenticList={};
    for Ext=1:length(ValidExtensions)
        SplicedList=[SplicedList;getAllFiles(SplicedPath,ValidExtensions{Ext},true)];
%         AuthenticList=[AuthenticList;getAllFiles(AuthenticPath,ValidExtensions{Ext},true)];
    end
%     warning('off','MATLAB:MKDIR:DirectoryExists');
    
    addpath(['.' filesep 'Algorithms' filesep AlgorithmName]);
    mkdir(strcat(Options.OutputPath,'Output_map'));
    for FileInd=1:length(SplicedList)
        OutputFile=[strrep(SplicedList{FileInd},SplicedPath,SplicedOutputPath) '.mat'];
        % If the .mat file already exists, skip it. This allows for partial
        % batch extraction. Remove if you intend to overwrite existing files
        if ~exist(OutputFile,'file')
            I = imread(SplicedList{FileInd});
            [row1,col1,~] = size(I);
            output_map = zeros(row1,col1);
            Result=analyze(SplicedList{FileInd});
            [row2,col2] = size(Result);
            if(strcmp(AlgorithmName,'ELA') || strcmp(AlgorithmName,'CAGI') || strcmp(AlgorithmName,'NOI4'))
                output_map = uint8(255*Result);
                tmp_name = strsplit(SplicedList{FileInd},'/');
                map_name = tmp_name{end};
                map_name = strrep(map_name,'.jpg','.png');
                map_name = strrep(map_name,'.tif','.png');
               
                imwrite(output_map,strcat(Options.OutputPath,'Output_map/',map_name));
            end
       
            if(strcmp(AlgorithmName,'NOI2'))
               step = 4;
               index_row = 1;
               index_col = 1;
               for i = 1:step:col1-step
                   for j = 1:step:row1-step
                       output_map(j:j+step-1,i:i+step-1) = Result(index_row,index_col);
                       index_row = index_row+1;
                   end
                   output_map(row1-step+1:row1,i:i+step-1) = Result(index_row,index_col);
                   index_col=index_col+1;
                   index_row = 1;
               end
               output_map(row1-step+1:row1,col1-step+1:col1) = Result(row2,col2);
               output_map = uint8(255*output_map);
               tmp_name = strsplit(SplicedList{FileInd},'/');
               map_name = tmp_name{end};
               map_name = strrep(map_name,'.jpg','.png');
               map_name = strrep(map_name,'.tif','.png');
               imwrite(output_map,strcat(Options.OutputPath,'Output_map/',map_name));
            end
             if(strcmp(AlgorithmName,'ADQ1') || strcmp(AlgorithmName,'ADQ2') || strcmp(AlgorithmName,'BLK')|| strcmp(AlgorithmName,'CFA1') || strcmp(AlgorithmName,'DCT') || strcmp(AlgorithmName,'NADQ'))
               step = 8;
               index_row = 1;
               index_col = 1;
               for i = 1:step:col1-step
                   for j = 1:step:row1-step
                       output_map(j:j+step-1,i:i+step-1) = Result(index_row,index_col);
                       index_row = index_row+1;
                   end
                   %index_row = index_row-1;
                   if(index_row>row2)
                       index_row = index_row-1;
                   end
                   
                   output_map(row1-step+1:row1,i:i+step-1) = Result(index_row,index_col);
                   index_col=index_col+1;
                   index_row = 1;
               end
               output_map(row1-step+1:row1,col1-step+1:col1) = Result(row2,col2);
               output_map = uint8(255*output_map);
               tmp_name = strsplit(SplicedList{FileInd},'\');
               map_name = tmp_name{end};
               map_name = strrep(map_name,'.jpg','.png');
               map_name = strrep(map_name,'.tif','.png');
               imwrite(output_map,strcat(Options.OutputPath,'Output_map/',map_name));
             end
            if(strcmp(AlgorithmName,'ADQ3') || strcmp(AlgorithmName,'CFA2') || strcmp(AlgorithmName,'NOI1'))
               step = 16;
               index_row = 1;
               index_col = 1;
               for i = 1:step:col1-step
                   for j = 1:step:row1-step
                       output_map(j:j+step-1,i:i+step-1) = Result(index_row,index_col);
                       index_row = index_row+1;
                   end
                   output_map(row1-step+1:row1,i:i+step-1) = Result(index_row,index_col);
                   index_col=index_col+1;
                   index_row = 1;
               end
               output_map(row1-step+1:row1,col1-step+1:col1) = Result(row2,col2);
               output_map = uint8(255*output_map);
               tmp_name = strsplit(SplicedList{FileInd},'\');
               map_name = tmp_name{end};
               map_name = strrep(map_name,'.jpg','.png');
               map_name = strrep(map_name,'.tif','.png');
               %map_name = strrep(map_name,'ps','ms');
               %map_name = strrep(map_name,'PS','MS');
               disp(tmp_name);
               imwrite(output_map,strcat(Options.OutputPath,'Output_map\',map_name));
            end
        end
%             [~,InputName,~]=fileparts(SplicedList{FileInd});
%             %one option is to have one mask per file with the same name and
%             %possibly different extension
%             BinMaskPath=dir([MasksPath InputName '.*']);
%             if ~isempty(BinMaskPath)
%                 Mask=mean(double(imread([MasksPath BinMaskPath.name])),3);
%                 MaskMin=min(Mask(:));
%                 MaskMax=max(Mask(:));
%                 MaskThresh=MaskMin+MaskMax/2;
%                 BinMask=Mask>MaskThresh;
%             else
%                 %the other is to have one mask in the entire folder, corresponding to
%                 %the entire dataset (such as the synthetic dataset of Fontani et al.)
%                 %make it a .png
%                 BinMaskPath=dir([MasksPath '*.png']);
%                 if length(BinMaskPath)>1
%                     error('Something is wrong with the masks');
%                 else
%                     Mask=mean(double(CleanUpImage([MasksPath BinMaskPath(1).name])),3);
%                     MaskMin=min(Mask(:));
%                     MaskMax=max(Mask(:));
%                     MaskThresh=MaskMin+MaskMax/2;
%                     BinMask=Mask>MaskThresh;
%                 end
%             end
%             [OutputPath,~,~]=fileparts(OutputFile);
%             mkdir(OutputPath);
%             save(OutputFile,'Result','AlgorithmName','BinMask','-v7.3');
%         end
    end
    
    % the ground truth mask for positive examples is taken from the root,
    % currently the square used in Fontani et al.
%     BinMask=mean(double(CleanUpImage('PositivesMask.png')),3)>128;
%     for FileInd=1:length(AuthenticList)
%         OutputFile=[strrep(AuthenticList{FileInd},AuthenticPath,AuthenticOutputPath) '.mat'];
%         % If the .mat file already exists, skip it. This allows for partial
%         % batch extraction. Remove if you intend to overwrite existing files
%         if ~exist(OutputFile,'file')
%             Result=analyze(AuthenticList{FileInd});
%             [Path,~,~]=fileparts(OutputFile);
%             mkdir(Path);
%             save(OutputFile,'Result','AlgorithmName','BinMask','-v7.3');
%         end
%     end
%     
%     warning('on','all');
%     rmpath(['.' filesep 'Algorithms' filesep AlgorithmName]);
end