% ART1.m 
%  Implement an ART (adpative resonance theory)
%  neural network to classify handwritten digits (unsupervised)
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

% FROM MOORE 1989
% Simple clustering algorithms are unstable
% "In ART1 plasticity is maintained because new clusters can be created at 
% any time.  The ART1 solution to the problem of instability is to
% constrain the prototype vectors to move only in certain directions in the
% update step."
% "The ART algorithm uses a first-order measure of similarity, and is not
% capable of finding higher-order information in the data.  Therefore it
% cannot perform shift-,scale-, or rotation-invariant classification on its
% inputs. These invariances can be achieved only by appropriately
% preprocessing the inputs."

load('TrainingData.mat')
numImages = size(Images,2);
Images(Images<=0.5) = 0;
Images(Images>0.5) = 1;
I = 1-Images;
Images = [Images;I];

LayerOne = 2*28*28; % call as myNet.BU or myNet.TD

vigVec = 1:6;
numCategories = zeros(length(vigVec),1);
runs = 100;
for vigilance = vigVec
    myNet = ART1Network(LayerOne);
    ImageIndeces = ceil(rand([runs,1]).*(numImages-1));
    for ii=1:runs
        index = ImageIndeces(ii);
        I = Images(:,index); %input
        T = myNet.TD;
        layerTwoSize = size(T,1);
        A = 1:layerTwoSize;
        
        while isempty(A) == 0
            s1 = (T*I)./(0.5+sum(T,2));
            tooFar = sum(I)/(0.5+LayerOne);
            [maxVal,maxInd] = max(s1);

            if maxVal <= tooFar
                myNet.TD = [myNet.TD;I'];
                break;
            else
                s2 = (T(maxInd,:)*I)/(sum(I));
                check = vigilance*0.1;
                if s2 < check
                    A = A([1:maxInd-1,maxInd+1:end]);
                    T = T([1:maxInd-1,maxInd+1:end],:);
                else
                    originalIndex = A(maxInd);
                    myNet.TD(originalIndex,:) = (T(maxInd,:)'.*I)';
                    break;
                end
            end
            
            if isempty(A) == 1
                myNet.TD = [myNet.TD;I'];
                break;
            end
        end
    end
    numCategories(vigilance) = size(myNet.TD,1);
end

figure();plot(vigVec.*0.1,numCategories);title('Number of ART Categories versus Vigilance Parameter, 1000 Training Epochs');
xlabel('Vigilance');ylabel('ART Categories');

figure();
count = 0;
randNums = randi([1,60000],[1,20]);
T = myNet.TD;
for ii=randNums
    count = count+1;
    I = Images(:,ii);
    s1 = (T*I)./(0.5+sum(T,2));
    [~,index] = max(s1);
    I = I(1:784);
    subplot(4,5,count);imagesc(reshape(I,[28,28]));title(sprintf('Cateogry %i',index));
    clear I;
end


        