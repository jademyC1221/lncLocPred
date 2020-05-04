function write4libsvm 

% Raw data is saved in the format: 
%             [label value1 value2...] 
% The format of the converted file meets the format requirements of libsvm, ie: 
%             [label 1:value1 2:value2 3:value3 ...] 
% Genial@ustc 
% 2004.6.16 
[filename, pathname] = uigetfile( {'*.mat', ... 
       '�����ļ�(*.mat)'; ... 
       '*.*',                   '�����ļ� (*.*)'}, ... 
   'ѡ�������ļ�'); 
try 
   S=load([pathname filename]); 
   fieldName = fieldnames(S); 
   str = cell2mat(fieldName); 
   B = getfield(S,str); 
   [m,n] = size(B); 
   [filename, pathname] = uiputfile({'*.txt;*.dat' ,'�����ļ�(*.txt;*.dat)';'*.*','�����ļ� (*.*)'},'���������ļ�'); 
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
       msgbox('�޷������ļ�!'); 
   end 
catch 
end