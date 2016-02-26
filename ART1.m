% ART.m 
%  Implement an ART (adpative resonance theory)
%  neural network to classify handwritten digits
%

% Carpenter, Grossberg 1987
% critical feature patterns - which are the critical features
% that make an object, or input, the same today as it was yesterday?
% The critical features of a pattern are separated from the noise in
% some way.
% 
% We have a sense of variability that is permissible, such that
% a given pattern of critical features is still that same pattern
% probably based on variations in our past experiences of that 
% pattern of features
%         Pick out the critical features of a pattern, distinct
%         from the noise.

load('TrainingData.mat')
numImages = size(Images,2);
Images(Images<=0.5) = 0;
Images(Images>0.5) = 1;


layerOne = 28*28; % call as myNet.BU or myNet.TD

vigVec = 1:9;
numCategories = zeros(length(vigVec),1);
runs = 10000;
parfor vigilance = vigVec
    myNet = ART1Network(layerOne);
    ImageIndeces = ceil(rand([runs,1]).*(numImages-1));
    for ii=1:runs
        index = ImageIndeces(ii);
        x = Images(:,index);
        B = myNet.BU;
        T = myNet.TD;
        layerTwoSize = size(B,2);
        A = 1:layerTwoSize;
        
        while isempty(A) == 0
            %N = sqrt(sum(abs(B).^2,1));
            Y = (B'*x); %.*N';
            R = tiedrank(Y);
            val = max(R);
            R(R<val) = 0;
            indeces = find(R);
            
            if length(indeces) == 1
                maxind = indeces;
            else
                randNums = randi([1,max(indeces)],[length(indeces),1]);
                [~,ind] = max(randNums);
                maxind = indeces(ind);
            end
            S = T(maxind,:)*x;
            sumx = sum(x);
            if S/sumx <= vigilance*0.1
                A = A([1:maxind-1,maxind+1:end]);
                B = B(:,[1:maxind-1,maxind+1:end]);
                T = T([1:maxind-1,maxind+1:end],:);
            else
                originalIndex = A(maxind);
                myNet.BU(:,originalIndex) = (T(maxind,:)'.*x)./(0.5+S);
                myNet.TD(originalIndex,:) = (T(maxind,:)'.*x)';
                break;
            end
            
            if isempty(A) == 1
                myNet.TD = [myNet.TD;x'];
                myNet.BU = [myNet.BU,zeros(layerOne,1)];
                myNet.BU(:,end) = (myNet.TD(end,:)'.*x)./(0.5+myNet.TD(end,:)*x);
                break;
            end
        end
    end
    numCategories(vigilance) = size(myNet.BU,2);
end

figure();plot(vigVec.*0.1,numCategories);title('Number of ART Categories versus Vigilance Parameter, 1000 Training Epochs');
xlabel('Vigilance');ylabel('ART Categories');

% for ii=[10,50,100,150,200,250]
%     x = Images(:,ii);
%     B = myNet.BU;
%     Y = (B'*x);
%     [~,index] = max(Y);
%     figure();imagesc(reshape(x,[28,28]));title(sprintf('Classified in cateogry %i',index));
% end


        