function  [kmernor,kmernorclorder]=KmernorCL112(filename,k)
%sequence to be read must be in fasta format.
dataS1 = fastaread(filename);
Seq = struct2cell(dataS1);
Seq = Seq(2,:)';
L=['A','C','G','T'];
kmernum=4.^k;
F=[];
for ii=1:kmernum
     E = [];
     r = ii-1;
     for jj=1:k
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
    for z = 1:kmernum
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
for i = 1:length(Seq)
    i
    W = [];
    for j = 1:kmernum
        f = kmer(i,j)/L1(i);
        W = [W f];
    end
   kmernor = [kmernor;W];
end
load CLorder.mat
%Clorder is the kmer feature order sorted by binomial distribution.
if k ==5
    CLorder = CLorder5;
elseif k==6
    CLorder = CLorder6;
elseif k==8
    CLorder = CLorder8;
end
kmernorclorder=[];
for j= 1:kmernum
    j
    E= kmernor(:,CLorder(j)); 
    kmernorclorder= [kmernorclorder E];
end
end