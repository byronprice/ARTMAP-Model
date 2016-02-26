function [myNet] = ART1Network(LayerOne)
% ART1Network.m
 % Define an 2-layer Network as a structure array
% INPUT: LayerVector - a vector, such as [784], with the number of
%         nodes per layer of the network.  ART networks always have 2
%         layers, and the second layer starts with only 1 node (or output)
% OUTPUT: Structure array with randomized weights and biases representing
%           the network.  Use standard normal random variables for initial
%           values.
% Created: 2016/02/25, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/02/25
%  By: Byron Price

field = 'BU';
field2 = 'TD';

value = cell(1,1);
value2 = cell(1,1);

value{1} = ones(LayerOne,1).*(1/(LayerOne+1));
value2{1} = ones(1,LayerOne);

myNet = struct(field,value,field2,value2);
end

