clc;clear all;close all;
addpath('./ompbox10');
data = load('DataBase.mat');  %load the provided data
DataBase = data.DataBase;
param.lambda =  0.002;
param.sparsity = 50;

%% This is a demo of SRC
lambda = param.lambda;
sparsity = param.sparsity;

tic                     % Start computing time from here
train = normc(DataBase.training_feats);
Y = DataBase.testing_feats;
W = DataBase.H_train;
test_label = DataBase.H_test;
Phi = train;
P = (Phi' * Phi + (lambda* eye(size(Phi,2))))\Phi' ;
A_check = P * Y;
G = Phi'*Phi;
A_hat = omp(Phi'*Y,G,sparsity);
Score = W * (normc((A_check) + (A_hat)));
Nt = size(Y,2);
tic
error_number=0;
number_class=38;
number_per_class=32;
SRC_A=A_hat;
for i=1:size(Y,2)
    SRC_A_each=SRC_A(:,i);
    test_sample=Y(:,i);
    error_each_class=zeros(number_class,1);
    score_gt = test_label(:,i);
    for j=1:number_class
        range=((j-1)*number_per_class+1:1:j*number_per_class);
        sub_dict=Phi(:,range);
        sub_coefficient= SRC_A_each(range,:);
        error_each_class(j)=sum(abs(test_sample-sub_dict*sub_coefficient));
    end
    [maxv, minind_est] = min(error_each_class);  
    [maxv, maxind_gt] = max(score_gt);
    if minind_est~=maxind_gt
        error_number=error_number+1;
    end          
end
accuracy_SRC = (Nt-error_number)/Nt *100;
time_SRC=toc;
st = ['Classification accuracy of SRC is ',  num2str(accuracy_SRC), '% in ' , num2str(time_SRC), ' seconds'];
disp (st)

