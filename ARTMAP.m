% ARTMAP.m 
%  Implement an ARTMAP (adpative resonance theory)
%  neural network to classify handwritten digits (unsupervised and supervised)
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
clear; 

load('TrainingData.mat')
numImages = size(Images,2);
Images(Images<=0.5) = 0;
Images(Images>0.5) = 1;
I = 1-Images;
Images = [Images;I];
clear I;

LayerOne = 2*28*28; % call as myNet.BU or myNet.TD
vigVec = 0.6;
runs = 10000;
beta = 0.8;
for vigilance = vigVec
    myNet = ARTMAPNetwork(LayerOne);
    ImageIndeces = ceil(rand([runs,1]).*(numImages-1));
    for ii=1:runs
        index = ImageIndeces(ii);
        I = Images(:,index); %input
        T = myNet.TD;
        layerTwoSize = size(T,1);
        
        s1 = (T*I)./(0.5+sum(T,2));
        tooFar = sum(I)/(0.5+LayerOne);
        [maxVal,~] = max(s1);
        indeces = find(s1==maxVal);
        randInt = randi([1,length(indeces)],1);
        maxInd = indeces(randInt);
        s2 = (T(maxInd,:)*I)/(sum(I));      

        if maxVal < tooFar || s2 < vigilance
            if ii == 1
                myNet.TD = I';
                myNet.MAP = [myNet.MAP;Labels(index)];
            else
                myNet.TD = [myNet.TD;I'];
                myNet.MAP = [myNet.MAP;Labels(index)];
            end
        elseif maxVal >= tooFar && s2 >= vigilance
            if ii == 1
                myNet.TD(maxInd,:) = beta.*(T(maxInd,:)'.*I)'+(1-beta).*T(maxInd,:);
                myNet.MAP = [myNet.MAP;Labels(index)];
            else
                myNet.TD(maxInd,:) = beta.*(T(maxInd,:)'.*I)'+(1-beta).*T(maxInd,:);
            end
        end
                
    end
end


clear Labels Images s1 s2;
load('TestData.mat');
numImages = size(Images,2);
Images(Images<=0.5) = 0;
Images(Images>0.5) = 1;
I = 1-Images;
Images = [Images;I];

count = 0;
T = myNet.TD;
M = myNet.MAP;
for ii=1:numImages
    truth = Labels(ii);
    s1 = (T*Images(:,ii))./(0.5+sum(T,2));
    [maxVal,~] = max(s1);
    indeces = find(s1==maxVal);
    randInt = randi([1,length(indeces)],1);
    index = indeces(randInt);
    output = M(index);
    if truth == output
        count = count+1;
    end
end
accuracy = count/numImages


        