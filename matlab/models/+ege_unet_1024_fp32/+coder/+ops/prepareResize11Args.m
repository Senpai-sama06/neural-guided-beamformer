function [DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, numDimsY] = prepareResize11Args(...
    ~, ONNXScales_, ONNXSizes_, coordinate_transformation_mode_, mode_, nearest_mode_, numDimsX_)
% Prepares arguments for implementing the ONNX Resize-11 operator
%#codegen

%   Copyright 2024-2025 The MathWorks, Inc.  

    ONNXScales                        = ege_unet_1024_fp32.coder.ops.extractIfDlarray(ONNXScales_);
    ONNXSizes                         = ege_unet_1024_fp32.coder.ops.extractIfDlarray(ONNXSizes_);
    coordinate_transformation_mode    = ege_unet_1024_fp32.coder.ops.extractIfDlarray(coordinate_transformation_mode_);
    mode                              = ege_unet_1024_fp32.coder.ops.extractIfDlarray(mode_);
    nearest_mode                      = ege_unet_1024_fp32.coder.ops.extractIfDlarray(nearest_mode_);
    numDimsX                          = ege_unet_1024_fp32.coder.ops.extractIfDlarray(numDimsX_);
    
    % ONNXScales and ONNXSizes are in ONNX dimension ordering. ONNXRoi is
    % ignored because it only takes effect when coordinate_transformation_mode
    % is "tf_crop_and_resize", which is not supported.
    DLTScales = flip(ONNXScales(:)');
    DLTSizes = flip(ONNXSizes(:)');
    
    switch coordinate_transformation_mode
        case "half_pixel"
            GeometricTransformMode = "half-pixel";
        case "asymmetric"
            GeometricTransformMode = "asymmetric";    
    end
    
    switch mode
        case "nearest"
            Method = "nearest";
        case "linear"
            Method = "linear";
    end
    
    switch nearest_mode
        case {"floor", "ceil", "round_prefer_floor"}
            NearestRoundingMode = "onnx-10";
        otherwise
            NearestRoundingMode = "round";
    end
    dataFormat = [repmat('S', [1 numDimsX-1]), 'B'];
    numDimsY = numDimsX;
end
