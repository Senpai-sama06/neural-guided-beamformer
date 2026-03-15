classdef Shape_To_ResizeLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end




    methods
        function this = Shape_To_ResizeLayer1000(name)
            this.Name = name;
            this.NumInputs = 2;
            this.OutputNames = {'x_gab1_Resize_output'};
        end

        function [x_gab1_Resize_output] = predict(this, x_enc1_relu_Relu_out, x_dec2_relu2_Relu_ou)
            if isdlarray(x_enc1_relu_Relu_out)
                x_enc1_relu_Relu_out = stripdims(x_enc1_relu_Relu_out);
            end
            if isdlarray(x_dec2_relu2_Relu_ou)
                x_dec2_relu2_Relu_ou = stripdims(x_dec2_relu2_Relu_ou);
            end
            x_enc1_relu_Relu_outNumDims = 4;
            x_dec2_relu2_Relu_ouNumDims = 4;
            x_enc1_relu_Relu_out = ege_unet_1024_fp32.ops.permuteInputVar(x_enc1_relu_Relu_out, [4 3 1 2], 4);
            x_dec2_relu2_Relu_ou = ege_unet_1024_fp32.ops.permuteInputVar(x_dec2_relu2_Relu_ou, [4 3 1 2], 4);

            [x_gab1_Resize_output, x_gab1_Resize_outputNumDims] = Shape_To_ResizeGraph1000(this, x_enc1_relu_Relu_out, x_dec2_relu2_Relu_ou, x_enc1_relu_Relu_outNumDims, x_dec2_relu2_Relu_ouNumDims, false);
            x_gab1_Resize_output = ege_unet_1024_fp32.ops.permuteOutputVar(x_gab1_Resize_output, [3 4 2 1], 4);

            x_gab1_Resize_output = dlarray(single(x_gab1_Resize_output), 'SSCB');
        end

        function [x_gab1_Resize_output] = forward(this, x_enc1_relu_Relu_out, x_dec2_relu2_Relu_ou)
            if isdlarray(x_enc1_relu_Relu_out)
                x_enc1_relu_Relu_out = stripdims(x_enc1_relu_Relu_out);
            end
            if isdlarray(x_dec2_relu2_Relu_ou)
                x_dec2_relu2_Relu_ou = stripdims(x_dec2_relu2_Relu_ou);
            end
            x_enc1_relu_Relu_outNumDims = 4;
            x_dec2_relu2_Relu_ouNumDims = 4;
            x_enc1_relu_Relu_out = ege_unet_1024_fp32.ops.permuteInputVar(x_enc1_relu_Relu_out, [4 3 1 2], 4);
            x_dec2_relu2_Relu_ou = ege_unet_1024_fp32.ops.permuteInputVar(x_dec2_relu2_Relu_ou, [4 3 1 2], 4);

            [x_gab1_Resize_output, x_gab1_Resize_outputNumDims] = Shape_To_ResizeGraph1000(this, x_enc1_relu_Relu_out, x_dec2_relu2_Relu_ou, x_enc1_relu_Relu_outNumDims, x_dec2_relu2_Relu_ouNumDims, true);
            x_gab1_Resize_output = ege_unet_1024_fp32.ops.permuteOutputVar(x_gab1_Resize_output, [3 4 2 1], 4);

            x_gab1_Resize_output = dlarray(single(x_gab1_Resize_output), 'SSCB');
        end

        function [x_gab1_Resize_output, x_gab1_Resize_outputNumDims1001] = Shape_To_ResizeGraph1000(this, x_enc1_relu_Relu_out, x_dec2_relu2_Relu_ou, x_enc1_relu_Relu_outNumDims, x_dec2_relu2_Relu_ouNumDims, Training)

            % Execute the operators:
            % Shape:
            [x_gab1_Shape_output_, x_gab1_Shape_output_NumDims] = ege_unet_1024_fp32.ops.onnxShape(x_enc1_relu_Relu_out, x_enc1_relu_Relu_outNumDims, 0, x_enc1_relu_Relu_outNumDims+1);

            % Gather:
            [x_gab1_Gather_output, x_gab1_Gather_outputNumDims] = ege_unet_1024_fp32.ops.onnxGather(x_gab1_Shape_output_, this.Vars.x_gab1_Constant_outp, 0, x_gab1_Shape_output_NumDims, this.NumDims.x_gab1_Constant_outp);

            % Shape:
            [x_gab1_Shape_1_outpu, x_gab1_Shape_1_outpuNumDims] = ege_unet_1024_fp32.ops.onnxShape(x_enc1_relu_Relu_out, x_enc1_relu_Relu_outNumDims, 0, x_enc1_relu_Relu_outNumDims+1);

            % Gather:
            [x_gab1_Gather_1_outp, x_gab1_Gather_1_outpNumDims] = ege_unet_1024_fp32.ops.onnxGather(x_gab1_Shape_1_outpu, this.Vars.x_gab1_Constant_1_ou, 0, x_gab1_Shape_1_outpuNumDims, this.NumDims.x_gab1_Constant_1_ou);

            % Unsqueeze:
            [shape, x_gab1_Unsqueeze_outNumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_gab1_Gather_output, this.Vars.onnx__Unsqueeze_453, x_gab1_Gather_outputNumDims);
            x_gab1_Unsqueeze_out = reshape(x_gab1_Gather_output, shape);

            % Unsqueeze:
            [shape, x_gab1_Unsqueeze_1_oNumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_gab1_Gather_1_outp, this.Vars.onnx__Unsqueeze_455, x_gab1_Gather_1_outpNumDims);
            x_gab1_Unsqueeze_1_o = reshape(x_gab1_Gather_1_outp, shape);

            % Concat:
            [x_gab1_Concat_output, x_gab1_Concat_outputNumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {x_gab1_Unsqueeze_out, x_gab1_Unsqueeze_1_o}, [x_gab1_Unsqueeze_outNumDims, x_gab1_Unsqueeze_1_oNumDims]);

            % Shape:
            [x_gab1_Shape_2_outpu, x_gab1_Shape_2_outpuNumDims] = ege_unet_1024_fp32.ops.onnxShape(x_dec2_relu2_Relu_ou, x_dec2_relu2_Relu_ouNumDims, 0, x_dec2_relu2_Relu_ouNumDims+1);

            % Slice:
            [Indices, x_gab1_Slice_output_NumDims] = ege_unet_1024_fp32.ops.prepareSliceArgs(x_gab1_Shape_2_outpu, this.Vars.x_gab1_Constant_3_ou, this.Vars.x_gab1_Constant_4_ou, this.Vars.x_gab1_Constant_2_ou, '', x_gab1_Shape_2_outpuNumDims);
            x_gab1_Slice_output_ = x_gab1_Shape_2_outpu(Indices{:});

            % Cast:
            x_gab1_Cast_output_0 = cast(int64(extractdata(x_gab1_Concat_output)), 'like', x_gab1_Concat_output);
            x_gab1_Cast_output_0NumDims = x_gab1_Concat_outputNumDims;

            % Concat:
            [x_gab1_Concat_1_outp, x_gab1_Concat_1_outpNumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {x_gab1_Slice_output_, x_gab1_Cast_output_0}, [x_gab1_Slice_output_NumDims, x_gab1_Cast_output_0NumDims]);

            % Resize:
            [DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, x_gab1_Resize_outputNumDims] = ege_unet_1024_fp32.ops.prepareResize11Args(dlarray([]), dlarray([]), x_gab1_Concat_1_outp, "half_pixel", "linear", "floor", x_dec2_relu2_Relu_ouNumDims);
            if isempty(DLTScales)
                x_gab1_Resize_output = dlresize(x_dec2_relu2_Relu_ou, 'OutputSize', DLTSizes, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            else
                x_gab1_Resize_output = dlresize(x_dec2_relu2_Relu_ou, 'Scale', DLTScales, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            end

            % Set graph output arguments
            x_gab1_Resize_outputNumDims1001 = x_gab1_Resize_outputNumDims;

        end

    end

end