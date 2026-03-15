function [DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, numDimsY] = prepareResize11Args(...
    ONNXRoi, ONNXScales, ONNXSizes, coordinate_transformation_mode, mode, nearest_mode, numDimsX)
% Prepares arguments for implementing the ONNX Resize-11 operator

%   Copyright 2020-2025 The MathWorks, Inc.    

% ONNXScales and ONNXSizes are in ONNX dimension ordering. ONNXRoi is
% ignored because it only takes effect when coordinate_transformation_mode
% is "tf_crop_and_resize", which is not supported.
if isdlarray(ONNXScales)
    ONNXScales = extractdata(ONNXScales);
end
if isdlarray(ONNXSizes)
    ONNXSizes = extractdata(ONNXSizes);
end

DLTScales = flip(ONNXScales(:)');
DLTSizes = flip(ONNXSizes(:)');
switch coordinate_transformation_mode
    case "half_pixel"
        GeometricTransformMode = "half-pixel";
    case "asymmetric"
        GeometricTransformMode = "asymmetric";
    otherwise
        assert(false);
end
switch mode
    case "nearest"
        Method = "nearest";
    case "linear"
        Method = "linear";
    otherwise
        assert(false);
end
switch nearest_mode
    case {"floor", "ceil","round_prefer_floor"}
        NearestRoundingMode = "onnx-10";
    otherwise
        NearestRoundingMode = "round";
end
dataFormat = repmat('S', [1 numDimsX]);
numDimsY = numDimsX;
end
