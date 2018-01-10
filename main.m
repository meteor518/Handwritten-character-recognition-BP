% clear
% clc
%% 读取样本，数字0-9每个2000幅,共20000组样本
feature = zeros(7,7);   %用于存储每幅图片的特征,以4*4分格子统计，得到49维特征
% 输入，15000组作为训练，5000组作为测试
X1 = zeros(15000,49);   %存储训练输入图像特征
X2 = zeros(5000,49);    %存储测试输入特征
% 理想输出 4维二进制表示
Y1 = zeros(15000,4);   %训练输出
Y2 = zeros(5000,4);    %测试输出
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
% 训练图像读取
X_index1 = 0;        %训练输入矩阵的行标
for num = 0:9
    numStr = num2str(num);
    trainpath = strcat('mnist\',numStr,'\');    %图像路径
    for image_num=0:1499                        %每个数字1500组数据作为训练
        I = imread(strcat(trainpath,num2str(image_num),'.bmp'));    %读取图像
        I = im2bw(I);    %转为二值化                               
        %以4*4进行划分，得到每个格的白色数，共49维特征
        for i=1:7
            for j=1:7
                sum77 = sum(I(((i*4-4+1):(i*4)),((j*4-4+1):(j*4))));
                feature(i,j)=sum(sum77);
            end
        end
        X_index1 = X_index1+1;
        X1(X_index1,:) = reshape(feature,1,49);  %将49维特征作为输入保存于X1
    end
    y = YY(num+1,:);
    for Y_index = X_index1-1500+1:X_index1     
        Y1(Y_index,:) = y;                       %保存已知输出
    end
end
% 测试图像读取
X_index2 = 0;        %训练输入矩阵的行标
for num = 0:9
    numStr = num2str(num);
    trainpath = strcat('mnist\',numStr,'\');
    for image_num=1500:1999
        I = imread(strcat(trainpath,num2str(image_num),'.bmp'));
        I = im2bw(I);
        %以7*7进行划分，得到每个格的白色数，共16维特征
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
%% 将20000组数据随机排序
k1 = rand(1,15000);
[value1,index1] = sort(k1);
k2 = rand(1,5000);
[value2,index2] = sort(k2);
input_train = X1(index1(1:15000),:)';
input_test = X2(index2(1:5000),:)';
output_train = Y1(index1(1:15000),:)';
output_test = Y2(index2(1:5000),:)';

%% 将输入数据归一化
[inputn,inputps] = mapminmax(input_train);

%% 初始化
% 网络结构初始化
innum = 49;
midnum = 49;
outnum = 4;

% 权值和阈值数组构建初始化
w1 = rands(midnum,innum);
b1 = rands(midnum,1);
w2 = rands(midnum,outnum);
b2 = rands(outnum,1);

w2_1 = w2;w2_2 = w2_1;
w1_1 = w1;w1_2 = w1_1;
b1_1 = b1;b1_2 = b1_1;
b2_1 = b2;b2_2 = b2_1;

yita = 0.1;             %学习速率
alpha = 0.01;           %惯性项
loop = 300;             %循环次数

% BP网络隐含层中间量数组
I = zeros(1,midnum);
Iout = zeros(1,midnum);
FI = zeros(1,midnum);

% 隐含层偏差及阈值求导量
dw1 = zeros(innum,midnum);
db1 = zeros(1,midnum);

%% 网络训练
E = zeros(1,loop);
for ii=1:loop
    E(ii)=0;
    for i=1:15000
        % 网络预测输出
        x = inputn(:,i);      
        % 隐含层输出
        for image_num=1:midnum
            I(image_num) = x'*w1(image_num,:)'+b1(image_num);
            Iout(image_num) = 1/(1+exp(-I(image_num)));
        end
        % 输出层输出
        yn = w2'*Iout'+b2;
        
       %% 权值阀值修正
        % 计算误差
        e = output_train(:,i)-yn;
        % 每次训练的累积误差
        E(ii) = E(ii)+sum(abs(e));
        % 计算权值变化
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

%% 测试
inputn_test=mapminmax('apply',input_test,inputps);
fore = zeros(4,5000);    %预测数据
%进行测试
for ii=1:1
    for i=1:5000%1500
        %隐含层输出
        for j=1:1:midnum
            I(j)=inputn_test(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        
        fore(:,i)=w2'*Iout'+b2;
    end
end
%% 结果分析
%根据网络输出找出数据属于哪类
output_fore = zeros(1,5000);
output1 = zeros(1,5000);
for i=1:5000
    output_fore(i) = round(fore(1,i))*8+round(fore(2,i))*4+round(fore(3,i))*2+round(fore(4,i))*1;
    output1(i) = round(output_test(1,i))*8+round(output_test(2,i))*4+round(output_test(3,i))*2+round(output_test(4,i))*1;
end

%BP网络预测误差
error = output_fore-output1;

%画出预测语音种类和实际语音种类的分类图
figure(1)
plot(output_fore,'r')
hold on
plot(output1,'b')
legend('预测数字','实际数字')

%画出误差图
figure(2)
plot(error)
title('BP网络分类误差','fontsize',12)
xlabel('手写数字','fontsize',12)
ylabel('分类误差','fontsize',12)

k = zeros(1,10);  
%找出判断错误的分类属于哪一类
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

%找出每类的个体和
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

%正确率
rightridio=(kk-k)./kk;
right = sum(rightridio)/length(rightridio);
disp('正确率')
disp(rightridio);
disp('平均正确率')
disp(right);