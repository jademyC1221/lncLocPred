function write4libsvm 

% Raw data is saved in the format: 
%             [label value1 value2...] 
% The format of the converted file meets the format requirements of libsvm, ie: 
%             [label 1:value1 2:value2 3:value3 ...] 
% Genial@ustc 
% 2004.6.16 
[filename, pathname] = uigetfile( {'*.mat', ... 
       '数据文件(*.mat)'; ... 
       '*.*',                   '所有文件 (*.*)'}, ... 
   '选择数据文件'); 
try 
   S=load([pathname filename]); 
   fieldName = fieldnames(S); 
   str = cell2mat(fieldName); 
   B = getfield(S,str); 
   [m,n] = size(B); 
   [filename, pathname] = uiputfile({'*.txt;*.dat' ,'数据文件(*.txt;*.dat)';'*.*','所有文件 (*.*)'},'保存数据文件'); 
   fid = fopen([pathname filename],'w'); 
   if(fid~=-1) 
       for k=1:m 
           fprintf(fid,'%3d',B(k,1)); 
           for kk = 2:n 
               fprintf(fid,'\t%d',(kk-1)); 
               fprintf(fid,':'); 
               fprintf(fid,'%d',B(k,kk)); 
           end 
           k
           fprintf(fid,'\n'); 
       end 
       fclose(fid); 
   else 
       msgbox('无法保存文件!'); 
   end 
catch 
end