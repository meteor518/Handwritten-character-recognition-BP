load('mnist_all.mat');
type = 'train';
savePath = 'F:\\mnist\';
for num = 0:1:9
    numStr = num2str(num);
    tempNumPath = strcat(savePath, numStr);
    mkdir(tempNumPath);
    tempNumPath = strcat(tempNumPath,'\');
    tempName = [type, numStr];
    tempFile = eval(tempName);
    [height, width]  = size(tempFile);
    for r = 1:1:height
        tempImg = reshape(tempFile(r,:),28,28)';
        tempImgPath = strcat(tempNumPath,num2str(r-1));
        tempImgPath = strcat(tempImgPath,'.bmp');
        imwrite(tempImg,tempImgPath);
    end
end