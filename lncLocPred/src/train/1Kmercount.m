function  [kmer,kmernor] = Kmercount(k)
Seq=Seqread();
L=['A','C','G','T'];
num = 4.^k;
F=[];
for ii=1:num;
     E = [];
     r = ii-1;
     for jj=1:k;
       A(jj)=L(mod(r,4)+1);
       r=floor(r/4);
     end
     E=A;
     F=[F;E];     
end

kmer = [];
for i = 1:length(Seq)
    i
    W = [];
    for z = 1:4.^k
        m = 0;
        for j = 1:(size(Seq{i,1},2)-(k-1))
            if strcmp(Seq{i,1}(1,j:j+(k-1)),F(z,1:k))
                m = m+1;
            end
        end
        X = m;
        W = [W X];
    end
    kmer = [kmer;W];
end

L1 = sum(kmer,2);            
kmernor = [];
for i = 1:size(kmer,1);
    i
    W = [];
    for j = 1:4.^k;
        f = kmer(i,j)/L1(i);  
        W = [W f];
    end
   kmernor = [kmernor;W];
end
end

function [Seq] = Seqread()
% benchmark dataset is on the supplementary material.
fid = fopen('S1 156nucleus.txt');
dataS1 = textscan(fid,'%s','delimiter','\n');
S1=dataS1{1,1}(2:2:end,:);          

fid = fopen('S2 426cytoplasm.txt');
dataS2 = textscan(fid,'%s','delimiter','\n');
S2=dataS2{1,1}(2:2:end,:);

fid = fopen('S3 43ribosome.txt');
dataS3 = textscan(fid,'%s','delimiter','\n');
S3=dataS3{1,1}(2:2:end,:);          

fid = fopen('S4 30exosome.txt');
dataS4 = textscan(fid,'%s','delimiter','\n');
S4=dataS4{1,1}(2:2:end,:); 

Seq=[S1;S2;S3;S4];
end