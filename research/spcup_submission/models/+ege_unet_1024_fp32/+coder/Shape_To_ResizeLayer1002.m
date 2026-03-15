classdef Shape_To_ResizeLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = ege_unet_1024_fp32.coder.Shape_To_ResizeLayer1002(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ege_unet_1024_fp32.Shape_To_ResizeLayer1002(cgInstance.Name);
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
        function this = Shape_To_ResizeLayer1002(mlInstance)
            this.Name = mlInstance.Name;
            this.NumInputs = 2;
            this.OutputNames = {'x_gab3_Resize_output'};
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

        function [x_gab3_Resize_output] = predict(this, x_enc3_res_relu2_Rel__, x_dec4_relu2_Relu_ou__)
            if isdlarray(x_enc3_res_relu2_Rel__)
                x_enc3_res_relu2_Rel_ = stripdims(x_enc3_res_relu2_Rel__);
            else
                x_enc3_res_relu2_Rel_ = x_enc3_res_relu2_Rel__;
            end
            if isdlarray(x_dec4_relu2_Relu_ou__)
                x_dec4_relu2_Relu_ou_ = stripdims(x_dec4_relu2_Relu_ou__);
            else
                x_dec4_relu2_Relu_ou_ = x_dec4_relu2_Relu_ou__;
            end
            x_enc3_res_relu2_RelNumDims = 4;
            x_dec4_relu2_Relu_ouNumDims = 4;
            x_enc3_res_relu2_Rel = ege_unet_1024_fp32.coder.ops.permuteInputVar(x_enc3_res_relu2_Rel_, [4 3 1 2], 4);
            x_dec4_relu2_Relu_ou = ege_unet_1024_fp32.coder.ops.permuteInputVar(x_dec4_relu2_Relu_ou_, [4 3 1 2], 4);

            [x_gab3_Resize_output__, x_gab3_Resize_outputNumDims__] = Shape_To_ResizeGraph1004(this, x_enc3_res_relu2_Rel, x_dec4_relu2_Relu_ou, x_enc3_res_relu2_RelNumDims, x_dec4_relu2_Relu_ouNumDims, false);
            x_gab3_Resize_output_ = ege_unet_1024_fp32.coder.ops.permuteOutputVar(x_gab3_Resize_output__, [3 4 2 1], 4);

            x_gab3_Resize_output = dlarray(single(x_gab3_Resize_output_), 'SSCB');
        end

        function [x_gab3_Resize_output, x_gab3_Resize_outputNumDims1005] = Shape_To_ResizeGraph1004(this, x_enc3_res_relu2_Rel, x_dec4_relu2_Relu_ou, x_enc3_res_relu2_RelNumDims, x_dec4_relu2_Relu_ouNumDims, Training)

            % Execute the operators:
            % Shape:
            [x_gab3_Shape_output_, x_gab3_Shape_output_NumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_enc3_res_relu2_Rel, coder.const(x_enc3_res_relu2_RelNumDims), 0, coder.const(x_enc3_res_relu2_RelNumDims)+1);

            % Gather:
            [x_gab3_Gather_output, x_gab3_Gather_outputNumDims] = ege_unet_1024_fp32.coder.ops.onnxGather(x_gab3_Shape_output_, this.Vars.x_gab3_Constant_outp, 0, coder.const(x_gab3_Shape_output_NumDims), this.NumDims.x_gab3_Constant_outp);

            % Shape:
            [x_gab3_Shape_1_outpu, x_gab3_Shape_1_outpuNumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_enc3_res_relu2_Rel, coder.const(x_enc3_res_relu2_RelNumDims), 0, coder.const(x_enc3_res_relu2_RelNumDims)+1);

            % Gather:
            [x_gab3_Gather_1_outp, x_gab3_Gather_1_outpNumDims] = ege_unet_1024_fp32.coder.ops.onnxGather(x_gab3_Shape_1_outpu, this.Vars.x_gab3_Constant_1_ou, 0, coder.const(x_gab3_Shape_1_outpuNumDims), this.NumDims.x_gab3_Constant_1_ou);

            % Unsqueeze:
            [shape1022, x_gab3_Unsqueeze_outNumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_gab3_Gather_output, this.Vars.onnx__Unsqueeze_373, coder.const(x_gab3_Gather_outputNumDims));
            x_gab3_Unsqueeze_out = reshape(x_gab3_Gather_output, shape1022);

            % Unsqueeze:
            [shape1023, x_gab3_Unsqueeze_1_oNumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_gab3_Gather_1_outp, this.Vars.onnx__Unsqueeze_375, coder.const(x_gab3_Gather_1_outpNumDims));
            x_gab3_Unsqueeze_1_o = reshape(x_gab3_Gather_1_outp, shape1023);

            % Concat:
            [x_gab3_Concat_output, x_gab3_Concat_outputNumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {x_gab3_Unsqueeze_out, x_gab3_Unsqueeze_1_o}, [coder.const(x_gab3_Unsqueeze_outNumDims), coder.const(x_gab3_Unsqueeze_1_oNumDims)]);

            % Shape:
            [x_gab3_Shape_2_outpu, x_gab3_Shape_2_outpuNumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_dec4_relu2_Relu_ou, coder.const(x_dec4_relu2_Relu_ouNumDims), 0, coder.const(x_dec4_relu2_Relu_ouNumDims)+1);

            % Slice:
            [indices1024, x_gab3_Slice_output_NumDims] = ege_unet_1024_fp32.coder.ops.prepareSliceArgs(x_gab3_Shape_2_outpu, this.Vars.x_gab3_Constant_3_ou, this.Vars.x_gab3_Constant_4_ou, this.Vars.x_gab3_Constant_2_ou, '', coder.const(x_gab3_Shape_2_outpuNumDims));
            x_gab3_Slice_output_ = x_gab3_Shape_2_outpu(indices1024{:});

            % Cast:
            x_gab3_Cast_output_0 = cast(int64(ege_unet_1024_fp32.coder.ops.extractIfDlarray(x_gab3_Concat_output)), 'like', x_gab3_Concat_output);
            x_gab3_Cast_output_0NumDims = coder.const(x_gab3_Concat_outputNumDims);

            % Concat:
            [x_gab3_Concat_1_outp, x_gab3_Concat_1_outpNumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {x_gab3_Slice_output_, x_gab3_Cast_output_0}, [coder.const(x_gab3_Slice_output_NumDims), coder.const(x_gab3_Cast_output_0NumDims)]);

            % Resize:
            [DLTScales1025, DLTSizes1026, dataFormat1027, Method1028, GeometricTransformMode1029, NearestRoundingMode1030, x_gab3_Resize_outputNumDims] = ege_unet_1024_fp32.coder.ops.prepareResize11Args(dlarray([]), dlarray([]), x_gab3_Concat_1_outp, "half_pixel", "linear", "floor", coder.const(x_dec4_relu2_Relu_ouNumDims));
            X1031 = dlarray(single(ege_unet_1024_fp32.coder.ops.extractIfDlarray(x_dec4_relu2_Relu_ou)));
            if isempty(DLTScales1025)
                Y1032 = dlresize(X1031, 'OutputSize', DLTSizes1026(1:end-1), 'DataFormat', dataFormat1027, 'Method', Method1028, 'GeometricTransformMode', GeometricTransformMode1029, 'NearestRoundingMode', NearestRoundingMode1030);
            else
                Y1032 = dlresize(X1031, 'Scale', DLTScales1025(1:end-1), 'DataFormat', dataFormat1027, 'Method', Method1028, 'GeometricTransformMode', GeometricTransformMode1029, 'NearestRoundingMode', NearestRoundingMode1030);
            end
            x_gab3_Resize_output = ege_unet_1024_fp32.coder.ops.extractIfDlarray(Y1032);

            % Set graph output arguments
            x_gab3_Resize_outputNumDims1005 = x_gab3_Resize_outputNumDims;

        end

    end

end