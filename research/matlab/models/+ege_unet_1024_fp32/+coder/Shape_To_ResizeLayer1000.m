classdef Shape_To_ResizeLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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

    methods(Static, Hidden)
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ege_unet_1024_fp32.coder.Shape_To_ResizeLayer1000(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ege_unet_1024_fp32.Shape_To_ResizeLayer1000(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
        end
    end

    methods
        function this = Shape_To_ResizeLayer1000(mlInstance)
            this.Name = mlInstance.Name;
            this.NumInputs = 2;
            this.OutputNames = {'x_gab1_Resize_output'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = ege_unet_1024_fp32.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
        end

        function [x_gab1_Resize_output] = predict(this, x_enc1_relu_Relu_out__, x_dec2_relu2_Relu_ou__)
            if isdlarray(x_enc1_relu_Relu_out__)
                x_enc1_relu_Relu_out_ = stripdims(x_enc1_relu_Relu_out__);
            else
                x_enc1_relu_Relu_out_ = x_enc1_relu_Relu_out__;
            end
            if isdlarray(x_dec2_relu2_Relu_ou__)
                x_dec2_relu2_Relu_ou_ = stripdims(x_dec2_relu2_Relu_ou__);
            else
                x_dec2_relu2_Relu_ou_ = x_dec2_relu2_Relu_ou__;
            end
            x_enc1_relu_Relu_outNumDims = 4;
            x_dec2_relu2_Relu_ouNumDims = 4;
            x_enc1_relu_Relu_out = ege_unet_1024_fp32.coder.ops.permuteInputVar(x_enc1_relu_Relu_out_, [4 3 1 2], 4);
            x_dec2_relu2_Relu_ou = ege_unet_1024_fp32.coder.ops.permuteInputVar(x_dec2_relu2_Relu_ou_, [4 3 1 2], 4);

            [x_gab1_Resize_output__, x_gab1_Resize_outputNumDims__] = Shape_To_ResizeGraph1000(this, x_enc1_relu_Relu_out, x_dec2_relu2_Relu_ou, x_enc1_relu_Relu_outNumDims, x_dec2_relu2_Relu_ouNumDims, false);
            x_gab1_Resize_output_ = ege_unet_1024_fp32.coder.ops.permuteOutputVar(x_gab1_Resize_output__, [3 4 2 1], 4);

            x_gab1_Resize_output = dlarray(single(x_gab1_Resize_output_), 'SSCB');
        end

        function [x_gab1_Resize_output, x_gab1_Resize_outputNumDims1001] = Shape_To_ResizeGraph1000(this, x_enc1_relu_Relu_out, x_dec2_relu2_Relu_ou, x_enc1_relu_Relu_outNumDims, x_dec2_relu2_Relu_ouNumDims, Training)

            % Execute the operators:
            % Shape:
            [x_gab1_Shape_output_, x_gab1_Shape_output_NumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_enc1_relu_Relu_out, coder.const(x_enc1_relu_Relu_outNumDims), 0, coder.const(x_enc1_relu_Relu_outNumDims)+1);

            % Gather:
            [x_gab1_Gather_output, x_gab1_Gather_outputNumDims] = ege_unet_1024_fp32.coder.ops.onnxGather(x_gab1_Shape_output_, this.Vars.x_gab1_Constant_outp, 0, coder.const(x_gab1_Shape_output_NumDims), this.NumDims.x_gab1_Constant_outp);

            % Shape:
            [x_gab1_Shape_1_outpu, x_gab1_Shape_1_outpuNumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_enc1_relu_Relu_out, coder.const(x_enc1_relu_Relu_outNumDims), 0, coder.const(x_enc1_relu_Relu_outNumDims)+1);

            % Gather:
            [x_gab1_Gather_1_outp, x_gab1_Gather_1_outpNumDims] = ege_unet_1024_fp32.coder.ops.onnxGather(x_gab1_Shape_1_outpu, this.Vars.x_gab1_Constant_1_ou, 0, coder.const(x_gab1_Shape_1_outpuNumDims), this.NumDims.x_gab1_Constant_1_ou);

            % Unsqueeze:
            [shape1000, x_gab1_Unsqueeze_outNumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_gab1_Gather_output, this.Vars.onnx__Unsqueeze_453, coder.const(x_gab1_Gather_outputNumDims));
            x_gab1_Unsqueeze_out = reshape(x_gab1_Gather_output, shape1000);

            % Unsqueeze:
            [shape1001, x_gab1_Unsqueeze_1_oNumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_gab1_Gather_1_outp, this.Vars.onnx__Unsqueeze_455, coder.const(x_gab1_Gather_1_outpNumDims));
            x_gab1_Unsqueeze_1_o = reshape(x_gab1_Gather_1_outp, shape1001);

            % Concat:
            [x_gab1_Concat_output, x_gab1_Concat_outputNumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {x_gab1_Unsqueeze_out, x_gab1_Unsqueeze_1_o}, [coder.const(x_gab1_Unsqueeze_outNumDims), coder.const(x_gab1_Unsqueeze_1_oNumDims)]);

            % Shape:
            [x_gab1_Shape_2_outpu, x_gab1_Shape_2_outpuNumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_dec2_relu2_Relu_ou, coder.const(x_dec2_relu2_Relu_ouNumDims), 0, coder.const(x_dec2_relu2_Relu_ouNumDims)+1);

            % Slice:
            [indices1002, x_gab1_Slice_output_NumDims] = ege_unet_1024_fp32.coder.ops.prepareSliceArgs(x_gab1_Shape_2_outpu, this.Vars.x_gab1_Constant_3_ou, this.Vars.x_gab1_Constant_4_ou, this.Vars.x_gab1_Constant_2_ou, '', coder.const(x_gab1_Shape_2_outpuNumDims));
            x_gab1_Slice_output_ = x_gab1_Shape_2_outpu(indices1002{:});

            % Cast:
            x_gab1_Cast_output_0 = cast(int64(ege_unet_1024_fp32.coder.ops.extractIfDlarray(x_gab1_Concat_output)), 'like', x_gab1_Concat_output);
            x_gab1_Cast_output_0NumDims = coder.const(x_gab1_Concat_outputNumDims);

            % Concat:
            [x_gab1_Concat_1_outp, x_gab1_Concat_1_outpNumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {x_gab1_Slice_output_, x_gab1_Cast_output_0}, [coder.const(x_gab1_Slice_output_NumDims), coder.const(x_gab1_Cast_output_0NumDims)]);

            % Resize:
            [DLTScales1003, DLTSizes1004, dataFormat1005, Method1006, GeometricTransformMode1007, NearestRoundingMode1008, x_gab1_Resize_outputNumDims] = ege_unet_1024_fp32.coder.ops.prepareResize11Args(dlarray([]), dlarray([]), x_gab1_Concat_1_outp, "half_pixel", "linear", "floor", coder.const(x_dec2_relu2_Relu_ouNumDims));
            X1009 = dlarray(single(ege_unet_1024_fp32.coder.ops.extractIfDlarray(x_dec2_relu2_Relu_ou)));
            if isempty(DLTScales1003)
                Y1010 = dlresize(X1009, 'OutputSize', DLTSizes1004(1:end-1), 'DataFormat', dataFormat1005, 'Method', Method1006, 'GeometricTransformMode', GeometricTransformMode1007, 'NearestRoundingMode', NearestRoundingMode1008);
            else
                Y1010 = dlresize(X1009, 'Scale', DLTScales1003(1:end-1), 'DataFormat', dataFormat1005, 'Method', Method1006, 'GeometricTransformMode', GeometricTransformMode1007, 'NearestRoundingMode', NearestRoundingMode1008);
            end
            x_gab1_Resize_output = ege_unet_1024_fp32.coder.ops.extractIfDlarray(Y1010);

            % Set graph output arguments
            x_gab1_Resize_outputNumDims1001 = x_gab1_Resize_outputNumDims;

        end

    end

end