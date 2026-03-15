classdef Shape_To_ResizeLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
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
        function this = Shape_To_ResizeLayer1002(name)
            this.Name = name;
            this.NumInputs = 2;
            this.OutputNames = {'x_gab3_Resize_output'};
        end

        function [x_gab3_Resize_output] = predict(this, x_enc3_res_relu2_Rel, x_dec4_relu2_Relu_ou)
            if isdlarray(x_enc3_res_relu2_Rel)
                x_enc3_res_relu2_Rel = stripdims(x_enc3_res_relu2_Rel);
            end
            if isdlarray(x_dec4_relu2_Relu_ou)
                x_dec4_relu2_Relu_ou = stripdims(x_dec4_relu2_Relu_ou);
            end
            x_enc3_res_relu2_RelNumDims = 4;
            x_dec4_relu2_Relu_ouNumDims = 4;
            x_enc3_res_relu2_Rel = ege_unet_1024_fp32.ops.permuteInputVar(x_enc3_res_relu2_Rel, [4 3 1 2], 4);
            x_dec4_relu2_Relu_ou = ege_unet_1024_fp32.ops.permuteInputVar(x_dec4_relu2_Relu_ou, [4 3 1 2], 4);

            [x_gab3_Resize_output, x_gab3_Resize_outputNumDims] = Shape_To_ResizeGraph1004(this, x_enc3_res_relu2_Rel, x_dec4_relu2_Relu_ou, x_enc3_res_relu2_RelNumDims, x_dec4_relu2_Relu_ouNumDims, false);
            x_gab3_Resize_output = ege_unet_1024_fp32.ops.permuteOutputVar(x_gab3_Resize_output, [3 4 2 1], 4);

            x_gab3_Resize_output = dlarray(single(x_gab3_Resize_output), 'SSCB');
        end

        function [x_gab3_Resize_output] = forward(this, x_enc3_res_relu2_Rel, x_dec4_relu2_Relu_ou)
            if isdlarray(x_enc3_res_relu2_Rel)
                x_enc3_res_relu2_Rel = stripdims(x_enc3_res_relu2_Rel);
            end
            if isdlarray(x_dec4_relu2_Relu_ou)
                x_dec4_relu2_Relu_ou = stripdims(x_dec4_relu2_Relu_ou);
            end
            x_enc3_res_relu2_RelNumDims = 4;
            x_dec4_relu2_Relu_ouNumDims = 4;
            x_enc3_res_relu2_Rel = ege_unet_1024_fp32.ops.permuteInputVar(x_enc3_res_relu2_Rel, [4 3 1 2], 4);
            x_dec4_relu2_Relu_ou = ege_unet_1024_fp32.ops.permuteInputVar(x_dec4_relu2_Relu_ou, [4 3 1 2], 4);

            [x_gab3_Resize_output, x_gab3_Resize_outputNumDims] = Shape_To_ResizeGraph1004(this, x_enc3_res_relu2_Rel, x_dec4_relu2_Relu_ou, x_enc3_res_relu2_RelNumDims, x_dec4_relu2_Relu_ouNumDims, true);
            x_gab3_Resize_output = ege_unet_1024_fp32.ops.permuteOutputVar(x_gab3_Resize_output, [3 4 2 1], 4);

            x_gab3_Resize_output = dlarray(single(x_gab3_Resize_output), 'SSCB');
        end

        function [x_gab3_Resize_output, x_gab3_Resize_outputNumDims1005] = Shape_To_ResizeGraph1004(this, x_enc3_res_relu2_Rel, x_dec4_relu2_Relu_ou, x_enc3_res_relu2_RelNumDims, x_dec4_relu2_Relu_ouNumDims, Training)

            % Execute the operators:
            % Shape:
            [x_gab3_Shape_output_, x_gab3_Shape_output_NumDims] = ege_unet_1024_fp32.ops.onnxShape(x_enc3_res_relu2_Rel, x_enc3_res_relu2_RelNumDims, 0, x_enc3_res_relu2_RelNumDims+1);

            % Gather:
            [x_gab3_Gather_output, x_gab3_Gather_outputNumDims] = ege_unet_1024_fp32.ops.onnxGather(x_gab3_Shape_output_, this.Vars.x_gab3_Constant_outp, 0, x_gab3_Shape_output_NumDims, this.NumDims.x_gab3_Constant_outp);

            % Shape:
            [x_gab3_Shape_1_outpu, x_gab3_Shape_1_outpuNumDims] = ege_unet_1024_fp32.ops.onnxShape(x_enc3_res_relu2_Rel, x_enc3_res_relu2_RelNumDims, 0, x_enc3_res_relu2_RelNumDims+1);

            % Gather:
            [x_gab3_Gather_1_outp, x_gab3_Gather_1_outpNumDims] = ege_unet_1024_fp32.ops.onnxGather(x_gab3_Shape_1_outpu, this.Vars.x_gab3_Constant_1_ou, 0, x_gab3_Shape_1_outpuNumDims, this.NumDims.x_gab3_Constant_1_ou);

            % Unsqueeze:
            [shape, x_gab3_Unsqueeze_outNumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_gab3_Gather_output, this.Vars.onnx__Unsqueeze_373, x_gab3_Gather_outputNumDims);
            x_gab3_Unsqueeze_out = reshape(x_gab3_Gather_output, shape);

            % Unsqueeze:
            [shape, x_gab3_Unsqueeze_1_oNumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_gab3_Gather_1_outp, this.Vars.onnx__Unsqueeze_375, x_gab3_Gather_1_outpNumDims);
            x_gab3_Unsqueeze_1_o = reshape(x_gab3_Gather_1_outp, shape);

            % Concat:
            [x_gab3_Concat_output, x_gab3_Concat_outputNumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {x_gab3_Unsqueeze_out, x_gab3_Unsqueeze_1_o}, [x_gab3_Unsqueeze_outNumDims, x_gab3_Unsqueeze_1_oNumDims]);

            % Shape:
            [x_gab3_Shape_2_outpu, x_gab3_Shape_2_outpuNumDims] = ege_unet_1024_fp32.ops.onnxShape(x_dec4_relu2_Relu_ou, x_dec4_relu2_Relu_ouNumDims, 0, x_dec4_relu2_Relu_ouNumDims+1);

            % Slice:
            [Indices, x_gab3_Slice_output_NumDims] = ege_unet_1024_fp32.ops.prepareSliceArgs(x_gab3_Shape_2_outpu, this.Vars.x_gab3_Constant_3_ou, this.Vars.x_gab3_Constant_4_ou, this.Vars.x_gab3_Constant_2_ou, '', x_gab3_Shape_2_outpuNumDims);
            x_gab3_Slice_output_ = x_gab3_Shape_2_outpu(Indices{:});

            % Cast:
            x_gab3_Cast_output_0 = cast(int64(extractdata(x_gab3_Concat_output)), 'like', x_gab3_Concat_output);
            x_gab3_Cast_output_0NumDims = x_gab3_Concat_outputNumDims;

            % Concat:
            [x_gab3_Concat_1_outp, x_gab3_Concat_1_outpNumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {x_gab3_Slice_output_, x_gab3_Cast_output_0}, [x_gab3_Slice_output_NumDims, x_gab3_Cast_output_0NumDims]);

            % Resize:
            [DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, x_gab3_Resize_outputNumDims] = ege_unet_1024_fp32.ops.prepareResize11Args(dlarray([]), dlarray([]), x_gab3_Concat_1_outp, "half_pixel", "linear", "floor", x_dec4_relu2_Relu_ouNumDims);
            if isempty(DLTScales)
                x_gab3_Resize_output = dlresize(x_dec4_relu2_Relu_ou, 'OutputSize', DLTSizes, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            else
                x_gab3_Resize_output = dlresize(x_dec4_relu2_Relu_ou, 'Scale', DLTScales, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            end

            % Set graph output arguments
            x_gab3_Resize_outputNumDims1005 = x_gab3_Resize_outputNumDims;

        end

    end

end