function [newShape, numDimsY] = prepareUnsqueezeArgs(X, ONNXAxes_, numDimsX_)
% Prepares arguments for implementing the ONNX Unsqueeze operator
%#codegen

%   Copyright 2024 The MathWorks, Inc.    

    ONNXAxes = ege_unet_1024_fp32.coder.ops.extractIfDlarray(ONNXAxes_);
    numDimsX = ege_unet_1024_fp32.coder.ops.extractIfDlarray(numDimsX_);
    
    numDimsY = numDimsX + numel(ONNXAxes);
    coder.unroll();
    for i = 1:numel(ONNXAxes)
        if ONNXAxes(i)<0
            ONNXAxes(i) = ONNXAxes(i) + numDimsY;
        end
    end 
    ONNXAxes = nnet.internal.cnn.coder.onnx.sort(ONNXAxes);               % increasing order
    
    if numDimsY == 1
        newShape = size(X);
    else
        DLTAxes  = flip(numDimsY - ONNXAxes);                                  % increasing order
        newShape = ones(1, numDimsY);
        posToSet = setdiff(1:numDimsY, DLTAxes, 'stable');
        newShape(posToSet) = size(X, 1:numel(posToSet));
    end
end
