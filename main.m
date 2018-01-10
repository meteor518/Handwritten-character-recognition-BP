% clear
% clc
%% ��ȡ����������0-9ÿ��2000��,��20000������
feature = zeros(7,7);   %���ڴ洢ÿ��ͼƬ������,��4*4�ָ���ͳ�ƣ��õ�49ά����
% ���룬15000����Ϊѵ����5000����Ϊ����
X1 = zeros(15000,49);   %�洢ѵ������ͼ������
X2 = zeros(5000,49);    %�洢������������
% ������� 4ά�����Ʊ�ʾ
Y1 = zeros(15000,4);   %ѵ�����
Y2 = zeros(5000,4);    %�������
YY = [0,0,0,0;...
    0,0,0,1;...
    0,0,1,0;...
    0,0,1,1;...
    0,1,0,0;...
    0,1,0,1;...
    0,1,1,0;...
    0,1,1,1;...
    1,0,0,0;...
    1,0,0,1;];
% YY = [1,0,0,0,0,0,0,0,0,0;...
%     0,1,0,0,0,0,0,0,0,0;...
%     0,0,1,0,0,0,0,0,0,0;...
%     0,0,0,1,0,0,0,0,0,0;...
%     0,0,0,0,1,0,0,0,0,0;...
%     0,0,0,0,0,1,0,0,0,0;...
%     0,0,0,0,0,0,1,0,0,0;...
%     0,0,0,0,0,0,0,1,0,0;...
%     0,0,0,0,0,0,0,0,1,0;...
%     0,0,0,0,0,0,0,0,0,1;];
% ѵ��ͼ���ȡ
X_index1 = 0;        %ѵ�����������б�
for num = 0:9
    numStr = num2str(num);
    trainpath = strcat('mnist\',numStr,'\');    %ͼ��·��
    for image_num=0:1499                        %ÿ������1500��������Ϊѵ��
        I = imread(strcat(trainpath,num2str(image_num),'.bmp'));    %��ȡͼ��
        I = im2bw(I);    %תΪ��ֵ��                               
        %��4*4���л��֣��õ�ÿ����İ�ɫ������49ά����
        for i=1:7
            for j=1:7
                sum77 = sum(I(((i*4-4+1):(i*4)),((j*4-4+1):(j*4))));
                feature(i,j)=sum(sum77);
            end
        end
        X_index1 = X_index1+1;
        X1(X_index1,:) = reshape(feature,1,49);  %��49ά������Ϊ���뱣����X1
    end
    y = YY(num+1,:);
    for Y_index = X_index1-1500+1:X_index1     
        Y1(Y_index,:) = y;                       %������֪���
    end
end
% ����ͼ���ȡ
X_index2 = 0;        %ѵ�����������б�
for num = 0:9
    numStr = num2str(num);
    trainpath = strcat('mnist\',numStr,'\');
    for image_num=1500:1999
        I = imread(strcat(trainpath,num2str(image_num),'.bmp'));
        I = im2bw(I);
        %��7*7���л��֣��õ�ÿ����İ�ɫ������16ά����
        for i=1:7
            for j=1:7
                sum77 = sum(I(((i*4-4+1):(i*4)),((j*4-4+1):(j*4))));
                feature(i,j)=sum(sum77);
            end
        end
        X_index2 = X_index2+1;
        X2(X_index2,:) = reshape(feature,1,49);
    end
    y = YY(num+1,:);
    for Y_index = X_index2-500+1:X_index2
        Y2(Y_index,:) = y;
    end
end
%% ��20000�������������
k1 = rand(1,15000);
[value1,index1] = sort(k1);
k2 = rand(1,5000);
[value2,index2] = sort(k2);
input_train = X1(index1(1:15000),:)';
input_test = X2(index2(1:5000),:)';
output_train = Y1(index1(1:15000),:)';
output_test = Y2(index2(1:5000),:)';

%% ���������ݹ�һ��
[inputn,inputps] = mapminmax(input_train);

%% ��ʼ��
% ����ṹ��ʼ��
innum = 49;
midnum = 49;
outnum = 4;

% Ȩֵ����ֵ���鹹����ʼ��
w1 = rands(midnum,innum);
b1 = rands(midnum,1);
w2 = rands(midnum,outnum);
b2 = rands(outnum,1);

w2_1 = w2;w2_2 = w2_1;
w1_1 = w1;w1_2 = w1_1;
b1_1 = b1;b1_2 = b1_1;
b2_1 = b2;b2_2 = b2_1;

yita = 0.1;             %ѧϰ����
alpha = 0.01;           %������
loop = 300;             %ѭ������

% BP�����������м�������
I = zeros(1,midnum);
Iout = zeros(1,midnum);
FI = zeros(1,midnum);

% ������ƫ���ֵ����
dw1 = zeros(innum,midnum);
db1 = zeros(1,midnum);

%% ����ѵ��
E = zeros(1,loop);
for ii=1:loop
    E(ii)=0;
    for i=1:15000
        % ����Ԥ�����
        x = inputn(:,i);      
        % ���������
        for image_num=1:midnum
            I(image_num) = x'*w1(image_num,:)'+b1(image_num);
            Iout(image_num) = 1/(1+exp(-I(image_num)));
        end
        % ��������
        yn = w2'*Iout'+b2;
        
       %% Ȩֵ��ֵ����
        % �������
        e = output_train(:,i)-yn;
        % ÿ��ѵ�����ۻ����
        E(ii) = E(ii)+sum(abs(e));
        % ����Ȩֵ�仯
        dw2 = e*Iout;
        db2 = e';
        for image_num=1:midnum
            S = 1/(1+exp(-I(image_num)));
            FI(image_num) = S*(1-S);
        end
        for k=1:innum
            for image_num=1:midnum
                dw1(k,image_num) = FI(image_num)*x(k)*(e(1)*w2(image_num,1)+e(2)*w2(image_num,2)+e(3)*w2(image_num,3)+e(4)*w2(image_num,4));
                db1(image_num) = FI(image_num)*(e(1)*w2(image_num,1)+e(2)*w2(image_num,2)+e(3)*w2(image_num,3)+e(4)*w2(image_num,4));
            end
        end
           
        w1 = w1_1+yita*dw1'+alpha*(w1_1-w1_2);
        b1 = b1_1+yita*db1'+alpha*(b1_1-b1_2);
        w2 = w2_1+yita*dw2'+alpha*(w2_1-w2_2);
        b2 = b2_1+yita*db2'+alpha*(b2_1-b2_2);
        
        w1_2 = w1_1;w1_1 = w1;
        w2_2 = w2_1;w2_1 = w2;
        b1_2 = b1_1;b1_1 = b1;
        b2_2 = b2_1;b2_1 = b2;
    end
end

%% ����
inputn_test=mapminmax('apply',input_test,inputps);
fore = zeros(4,5000);    %Ԥ������
%���в���
for ii=1:1
    for i=1:5000%1500
        %���������
        for j=1:1:midnum
            I(j)=inputn_test(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        
        fore(:,i)=w2'*Iout'+b2;
    end
end
%% �������
%������������ҳ�������������
output_fore = zeros(1,5000);
output1 = zeros(1,5000);
for i=1:5000
    output_fore(i) = round(fore(1,i))*8+round(fore(2,i))*4+round(fore(3,i))*2+round(fore(4,i))*1;
    output1(i) = round(output_test(1,i))*8+round(output_test(2,i))*4+round(output_test(3,i))*2+round(output_test(4,i))*1;
end

%BP����Ԥ�����
error = output_fore-output1;

%����Ԥ�����������ʵ����������ķ���ͼ
figure(1)
plot(output_fore,'r')
hold on
plot(output1,'b')
legend('Ԥ������','ʵ������')

%�������ͼ
figure(2)
plot(error)
title('BP����������','fontsize',12)
xlabel('��д����','fontsize',12)
ylabel('�������','fontsize',12)

k = zeros(1,10);  
%�ҳ��жϴ���ķ���������һ��
for i=1:5000
    if error(i)~=0
        c = output1(:,i);
        switch c
            case 0 
                k(1)=k(1)+1;
            case 1 
                k(2)=k(2)+1;
            case 2 
                k(3)=k(3)+1;
            case 3 
                k(4)=k(4)+1;
            case 4 
                k(5)=k(5)+1;
            case 5 
                k(6)=k(6)+1;
            case 6 
                k(7)=k(7)+1;
            case 7 
                k(8)=k(8)+1;
            case 8
                k(9)=k(9)+1;
            case 9
                k(10)=k(10)+1;
        end
    end
end

%�ҳ�ÿ��ĸ����
kk = zeros(1,10);
for i=1:5000
    c = output1(:,i);
    switch c
        case 0 
            kk(1)=kk(1)+1;
        case 1 
            kk(2)=kk(2)+1;
        case 2 
            kk(3)=kk(3)+1;
        case 3 
            kk(4)=kk(4)+1;
        case 4 
            kk(5)=kk(5)+1;
        case 5 
            kk(6)=kk(6)+1;
        case 6 
            kk(7)=kk(7)+1;
        case 7 
            kk(8)=kk(8)+1;
        case 8
            kk(9)=kk(9)+1;
        case 9
            kk(10)=kk(10)+1;
    end
end

%��ȷ��
rightridio=(kk-k)./kk;
right = sum(rightridio)/length(rightridio);
disp('��ȷ��')
disp(rightridio);
disp('ƽ����ȷ��')
disp(right);