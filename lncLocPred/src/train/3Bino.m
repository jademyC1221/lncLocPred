function kmerCLorder=Bino(k)
% k is the type of kmer.
load lncRNAkmer.mat
%lncRNAkmer.mat contains the kmer features processed by the KMER_VarianceThreshold.py.

if k==5:
    m1=sum(sum(lncRNA5mer655(1:156,:)));                
    m2=sum(sum(lncRNA5mer655(157:582,:)));
    m3=sum(sum(lncRNA5mer655(583:625,:)));
    m4=sum(sum(lncRNA5mer655(626:655,:)));
    M=m1+m2+m3+m4;              
    q1=m1/M;
    q2=m2/M;
    q3=m3/M;
    q4=m4/M;
    Q = [q1 q2 q3 q4];
    ni1 = sum(lncRNA5mer655(1:156,:));
    ni2 = sum(lncRNA5mer655(157:582,:));
    ni3 = sum(lncRNA5mer655(583:625,:));
    ni4 = sum(lncRNA5mer655(626:655,:));
    W = [ni1;ni2;ni3;ni4];
    W = W';
    G = sum(lncRNA5mer655);
    PP=[];
    for i=1:1024
        i
       P=[];
       for j=1:4
         sum = 0;
         for m=W(i,j):G(i)
              sum = binopdf(m,G(i),Q(j))+sum;
         end
         E = sum;
         P = [P E];
        end
        PP = [PP;P];
    end
    CL = 1-PP;                    
    [max_CL,index]=max(CL,[],2);           
    CLi = [max_CL,index];
    CLimax = CLi(:,1);                    
    Feorder = (1:1024)';                 
    CLimax_order = [CLimax,Feorder];     
    CLimax_order = sortrows(CLimax_order,-1);   
    CLorder5 = CLimax_order(:,2);
    lnc5mernor655CL=[];
    for j= 1:1024
        j
        E= lnc5mer655nor(:,CLorder5(j)); 
        lnc5mernor655CL= [lnc5mernor655CL E];                
    end
    kmerCLorder=lnc5mernor655CL;
elseif k==6:
    m1=sum(sum(lncRNA6mer655(1:156,:)));                
    m2=sum(sum(lncRNA6mer655(157:582,:)));
    m3=sum(sum(lncRNA6mer655(583:625,:)));
    m4=sum(sum(lncRNA6mer655(626:655,:)));
    M=m1+m2+m3+m4;              
    q1=m1/M;
    q2=m2/M;
    q3=m3/M;
    q4=m4/M;
    Q = [q1 q2 q3 q4];
    ni1 = sum(lncRNA6mer655(1:156,:));
    ni2 = sum(lncRNA6mer655(157:582,:));
    ni3 = sum(lncRNA6mer655(583:625,:));
    ni4 = sum(lncRNA6mer655(626:655,:));
    W = [ni1;ni2;ni3;ni4];
    W = W';
    G = sum(lncRNA6mer655);
    PP=[];
    for i=1:4096
        i
       P=[];
       for j=1:4
         sum = 0;
         for m=W(i,j):G(i)
              sum = binopdf(m,G(i),Q(j))+sum;
         end
         E = sum;
         P = [P E];
        end
        PP = [PP;P];
    end
    CL = 1-PP;                    
    [max_CL,index]=max(CL,[],2);           
    CLi = [max_CL,index];
    CLimax = CLi(:,1);                    
    Feorder = (1:4096)';                 
    CLimax_order = [CLimax,Feorder];     
    CLimax_order = sortrows(CLimax_order,-1);   
    CLorder6 = CLimax_order(:,2);
    lnc6mernor655CL=[];
    for j= 1:4096
        j
        E= lnc6mer655nor(:,CLorder6(j)); 
        lnc6mernor655CL= [lnc6mernor655CL E];                
    end
    kmerCLorder=lnc6mernor655CL;
elseif k==8:
    m1=sum(sum(lncRNA8mer655_2(1:156,:)));                
    m2=sum(sum(lncRNA8mer655_2(157:582,:)));
    m3=sum(sum(lncRNA8mer655_2(583:625,:)));
    m4=sum(sum(lncRNA8mer655_2(626:655,:)));
    M=m1+m2+m3+m4;              
    q1=m1/M;
    q2=m2/M;
    q3=m3/M;
    q4=m4/M;
    Q = [q1 q2 q3 q4];
    ni1 = sum(lncRNA8mer655_2(1:156,:));
    ni2 = sum(lncRNA8mer655_2(157:582,:));
    ni3 = sum(lncRNA8mer655_2(583:625,:));
    ni4 = sum(lncRNA8mer655_2(626:655,:));
    W = [ni1;ni2;ni3;ni4];
    W = W';
    G = sum(lncRNA8mer655_2);
    PP=[];
    for i=1:65536
        i
       P=[];
       for j=1:4
         sum = 0;
         for m=W(i,j):G(i)
              sum = binopdf(m,G(i),Q(j))+sum;
         end
         E = sum;
         P = [P E];
        end
        PP = [PP;P];
    end
    CL = 1-PP;                    
    [max_CL,index]=max(CL,[],2);           
    CLi = [max_CL,index];
    CLimax = CLi(:,1);                    
    Feorder = (1:65536)';                 
    CLimax_order = [CLimax,Feorder];     
    CLimax_order = sortrows(CLimax_order,-1);   
    CLorder8 = CLimax_order(:,2);
    
    index = xlsread('octamerindex.xlsx');
    lnc8mernor655_2=[];jj=1;
    for ii=1:65536
       if index(ii)==1
           lnc8mernor655_2(:,jj) = lnc8mer655nor(:,ii);
           jj=jj+1;
       end
    end
    for j= 1:size(lnc8mernor655_2,2)
        j
        E= lnc8mernor655_2(:,CLorder8(j)); 
        lnc8mernor655CL = [lnc8mernor655CL E];
    end    
    kmerCLorder=lnc8mernor655CL;
    
end
    
end