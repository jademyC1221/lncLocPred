
%PseDNC.mat is the pseDNC feature with optimized parameters
%Triplet.mat is the triplet feature of benchmark dataset calculated from http://bioinformatics.hitsz.edu.cn/Pse-in-One2.0/
%num5mer,num6mer,num8mer are the seleted feature number of 5mer,6mer and 8mer after IFS.py.
%take 3 kinds of feature combinations(tp5,tp6,tp8) as example.
load PseDNC.mat
load Triplet.mat
load lncRNAkmer.mat
load label.mat
mer5 = lnc5mernor655CL(:,1:num5mer);
mer6 = lnc6mernor655CL(:,1:num6mer);
mer8 = lnc8mernor655CL(:,1:num8mer);
fscpertp5 = [label Triplet PseDNC mer5];
pertp5 = [Triplet PseDNC mer5];
fscpertp6 = [label Triplet PseDNC mer6];
pertp6 = [Triplet PseDNC mer6];
fscpertp8 = [label Triplet PseDNC mer8];
pertp8 = [Triplet PseDNC mer8];
fscpertp56 = [label Triplet PseDNC mer5 mer6];
pertp56 = [Triplet PseDNC mer5 mer6];
fscpertp58 = [label Triplet PseDNC mer5 mer8];
pertp58 = [Triplet PseDNC mer5 mer8];
fscpertp68 = [label Triplet PseDNC mer6 mer8];
pertp68 = [Triplet PseDNC mer6 mer8];
fscpertp568 = [label Triplet PseDNC mer5 mer6 mer8];
pertp568 = [Triplet PseDNC mer5 mer6 mer8];

%%save tpk.mat as the input of write4libsvm.m to obtain£ºtpkfsc.txt
%put fselect.py and fscpertpk.txt on the same folder
%path£¬and run "python fselect.py fscpertpk.txt"

A1=textread('fscpertp5.txt.fscore','%s','delimiter',':');
A2=textread('fscpertp6.txt.fscore','%s','delimiter',':');
A3=textread('fscpertp8.txt.fscore','%s','delimiter',':');
A4=textread('fscpertp56.txt.fscore','%s','delimiter',':');
A5=textread('fscpertp58.txt.fscore','%s','delimiter',':');
A6=textread('fscpertp68.txt.fscore','%s','delimiter',':');
A7=textread('fscpertp568.txt.fscore','%s','delimiter',':');

% take A3 as an example
h=size(A3,1)/2-1;     
FscoreS = [];
for i = 0:h
    i
    D11 = A3(2*i+1,:);
    FscoreS = [FscoreS D11];
end
FscoreS = str2double(FscoreS);
pertp8fsc=[];
for j= 1:size(pertp8,2)
    j
    E= pertp8(:,FscoreS(j));
    pertp8fsc= [pertp8fsc E];
end