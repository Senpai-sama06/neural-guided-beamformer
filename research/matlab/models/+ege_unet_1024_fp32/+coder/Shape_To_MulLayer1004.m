classdef Shape_To_MulLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        onnx__Concat_556
        onnx__Resize_554
        onnx__Resize_557
        onnx__Resize_560
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
                'onnx__Concat_556'
                'onnx__Resize_554'
                'onnx__Resize_557'
                'onnx__Resize_560'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ege_unet_1024_fp32.coder.Shape_To_MulLayer1004(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ege_unet_1024_fp32.Shape_To_MulLayer1004(cgInstance.Name);
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
            this_ml.onnx__Concat_556 = cgInstance.onnx__Concat_556;
            this_ml.onnx__Resize_554 = cgInstance.onnx__Resize_554;
            this_ml.onnx__Resize_557 = cgInstance.onnx__Resize_557;
            this_ml.onnx__Resize_560 = cgInstance.onnx__Resize_560;
        end
    end

    methods
        function this = Shape_To_MulLayer1004(mlInstance)
            this.Name = mlInstance.Name;
            this.NumOutputs = 4;
            this.OutputNames = {'x_bottleneck_Mul_4_o', 'x_bottleneck_Mul_5_o', 'x_bottleneck_Mul_6_o', 'x_bottleneck_Slice_3'};
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
            this.onnx__Concat_556 = mlInstance.onnx__Concat_556;
            this.onnx__Resize_554 = mlInstance.onnx__Resize_554;
            this.onnx__Resize_557 = mlInstance.onnx__Resize_557;
            this.onnx__Resize_560 = mlInstance.onnx__Resize_560;
        end

        function [x_bottleneck_Mul_4_o, x_bottleneck_Mul_5_o, x_bottleneck_Mul_6_o, x_bottleneck_Slice_3] = predict(this, x_pool_3_MaxPool_out__)
            if isdlarray(x_pool_3_MaxPool_out__)
                x_pool_3_MaxPool_out_ = stripdims(x_pool_3_MaxPool_out__);
            else
                x_pool_3_MaxPool_out_ = x_pool_3_MaxPool_out__;
            end
            x_pool_3_MaxPool_outNumDims = 4;
            x_pool_3_MaxPool_out = ege_unet_1024_fp32.coder.ops.permuteInputVar(x_pool_3_MaxPool_out_, [4 3 1 2], 4);

            [x_bottleneck_Mul_4_o__, x_bottleneck_Mul_5_o__, x_bottleneck_Mul_6_o__, x_bottleneck_Slice_3__, x_bottleneck_Mul_4_oNumDims__, x_bottleneck_Mul_5_oNumDims__, x_bottleneck_Mul_6_oNumDims__, x_bottleneck_Slice_3NumDims__] = Shape_To_MulGraph1008(this, x_pool_3_MaxPool_out, x_pool_3_MaxPool_outNumDims, false);
            x_bottleneck_Mul_4_o_ = ege_unet_1024_fp32.coder.ops.permuteOutputVar(x_bottleneck_Mul_4_o__, [3 4 2 1], 4);
            x_bottleneck_Mul_5_o_ = ege_unet_1024_fp32.coder.ops.permuteOutputVar(x_bottleneck_Mul_5_o__, [3 4 2 1], 4);
            x_bottleneck_Mul_6_o_ = ege_unet_1024_fp32.coder.ops.permuteOutputVar(x_bottleneck_Mul_6_o__, [3 4 2 1], 4);
            x_bottleneck_Slice_3_ = ege_unet_1024_fp32.coder.ops.permuteOutputVar(x_bottleneck_Slice_3__, [3 4 2 1], 4);

            x_bottleneck_Mul_4_o = dlarray(single(x_bottleneck_Mul_4_o_), 'SSCB');
            x_bottleneck_Mul_5_o = dlarray(single(x_bottleneck_Mul_5_o_), 'SSCB');
            x_bottleneck_Mul_6_o = dlarray(single(x_bottleneck_Mul_6_o_), 'SSCB');
            x_bottleneck_Slice_3 = dlarray(single(x_bottleneck_Slice_3_), 'SSCB');
        end

        function [x_bottleneck_Mul_4_o, x_bottleneck_Mul_5_o, x_bottleneck_Mul_6_o, x_bottleneck_Slice_3, x_bottleneck_Mul_4_oNumDims1009, x_bottleneck_Mul_5_oNumDims1010, x_bottleneck_Mul_6_oNumDims1011, x_bottleneck_Slice_3NumDims1012] = Shape_To_MulGraph1008(this, x_pool_3_MaxPool_out, x_pool_3_MaxPool_outNumDims, Training)

            % Execute the operators:
            % Identity:
            onnx__Concat_562 = this.onnx__Concat_556;
            onnx__Concat_562NumDims = this.NumDims.onnx__Concat_556;

            % Identity:
            onnx__Concat_559 = this.onnx__Concat_556;
            onnx__Concat_559NumDims = this.NumDims.onnx__Concat_556;

            % Shape:
            [x_bottleneck_Shape_o, x_bottleneck_Shape_oNumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_pool_3_MaxPool_out, coder.const(x_pool_3_MaxPool_outNumDims), 0, coder.const(x_pool_3_MaxPool_outNumDims)+1);

            % Gather:
            [x_bottleneck_Gathe_2, x_bottleneck_Gathe_2NumDims] = ege_unet_1024_fp32.coder.ops.onnxGather(x_bottleneck_Shape_o, this.Vars.x_bottleneck_Cons_9, 0, coder.const(x_bottleneck_Shape_oNumDims), this.NumDims.x_bottleneck_Cons_9);

            % Shape:
            [x_bottleneck_Shape_1, x_bottleneck_Shape_1NumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_pool_3_MaxPool_out, coder.const(x_pool_3_MaxPool_outNumDims), 0, coder.const(x_pool_3_MaxPool_outNumDims)+1);

            % Gather:
            [x_bottleneck_Gather_, x_bottleneck_Gather_NumDims] = ege_unet_1024_fp32.coder.ops.onnxGather(x_bottleneck_Shape_1, this.Vars.x_bottleneck_Constan, 0, coder.const(x_bottleneck_Shape_1NumDims), this.NumDims.x_bottleneck_Constan);

            % Shape:
            [x_bottleneck_Shape_2, x_bottleneck_Shape_2NumDims] = ege_unet_1024_fp32.coder.ops.onnxShape(x_pool_3_MaxPool_out, coder.const(x_pool_3_MaxPool_outNumDims), 0, coder.const(x_pool_3_MaxPool_outNumDims)+1);

            % Gather:
            [x_bottleneck_Gathe_1, x_bottleneck_Gathe_1NumDims] = ege_unet_1024_fp32.coder.ops.onnxGather(x_bottleneck_Shape_2, this.Vars.x_bottleneck_Cons_1, 0, coder.const(x_bottleneck_Shape_2NumDims), this.NumDims.x_bottleneck_Cons_1);

            % Add:
            x_bottleneck_Add_out = x_bottleneck_Gathe_1 + this.Vars.x_bottleneck_Cons_3;
            x_bottleneck_Add_outNumDims = max(coder.const(x_bottleneck_Gathe_1NumDims), this.NumDims.x_bottleneck_Cons_3);

            % Div:
            x_bottleneck_Div_out = fix(x_bottleneck_Add_out ./ this.Vars.x_bottleneck_Cons_4);
            x_bottleneck_Div_outNumDims = max(coder.const(x_bottleneck_Add_outNumDims), this.NumDims.x_bottleneck_Cons_4);

            % Mul:
            x_bottleneck_Mul_out = x_bottleneck_Div_out .* this.Vars.x_bottleneck_Cons_5;
            x_bottleneck_Mul_outNumDims = max(coder.const(x_bottleneck_Div_outNumDims), this.NumDims.x_bottleneck_Cons_5);

            % Slice:
            [indices1044, x_bottleneck_Slice_oNumDims] = ege_unet_1024_fp32.coder.ops.prepareSliceArgs(x_pool_3_MaxPool_out, this.Vars.x_bottleneck_Cons_2, x_bottleneck_Mul_out, this.Vars.x_bottleneck_Cons_1, '', coder.const(x_pool_3_MaxPool_outNumDims));
            x_bottleneck_Slice_o = x_pool_3_MaxPool_out(indices1044{:});

            % Mul:
            x_bottleneck_Mul_1_o = x_bottleneck_Div_out .* this.Vars.x_bottleneck_Cons_6;
            x_bottleneck_Mul_1_oNumDims = max(coder.const(x_bottleneck_Div_outNumDims), this.NumDims.x_bottleneck_Cons_6);

            % Slice:
            [indices1045, x_bottleneck_Slice_1NumDims] = ege_unet_1024_fp32.coder.ops.prepareSliceArgs(x_pool_3_MaxPool_out, x_bottleneck_Mul_out, x_bottleneck_Mul_1_o, this.Vars.x_bottleneck_Cons_1, '', coder.const(x_pool_3_MaxPool_outNumDims));
            x_bottleneck_Slice_1 = x_pool_3_MaxPool_out(indices1045{:});

            % Mul:
            x_bottleneck_Mul_2_o = x_bottleneck_Div_out .* this.Vars.x_bottleneck_Cons_7;
            x_bottleneck_Mul_2_oNumDims = max(coder.const(x_bottleneck_Div_outNumDims), this.NumDims.x_bottleneck_Cons_7);

            % Slice:
            [indices1046, x_bottleneck_Slice_2NumDims] = ege_unet_1024_fp32.coder.ops.prepareSliceArgs(x_pool_3_MaxPool_out, x_bottleneck_Mul_1_o, x_bottleneck_Mul_2_o, this.Vars.x_bottleneck_Cons_1, '', coder.const(x_pool_3_MaxPool_outNumDims));
            x_bottleneck_Slice_2 = x_pool_3_MaxPool_out(indices1046{:});

            % Mul:
            x_bottleneck_Mul_3_o = x_bottleneck_Div_out .* this.Vars.x_bottleneck_Cons_8;
            x_bottleneck_Mul_3_oNumDims = max(coder.const(x_bottleneck_Div_outNumDims), this.NumDims.x_bottleneck_Cons_8);

            % Slice:
            [indices1047, x_bottleneck_Slice_3NumDims] = ege_unet_1024_fp32.coder.ops.prepareSliceArgs(x_pool_3_MaxPool_out, x_bottleneck_Mul_2_o, x_bottleneck_Mul_3_o, this.Vars.x_bottleneck_Cons_1, '', coder.const(x_pool_3_MaxPool_outNumDims));
            x_bottleneck_Slice_3 = x_pool_3_MaxPool_out(indices1047{:});

            % Unsqueeze:
            [shape1048, x_bottleneck_Unsq_5NumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_bottleneck_Gathe_2, this.Vars.onnx__Unsqueeze_274, coder.const(x_bottleneck_Gathe_2NumDims));
            x_bottleneck_Unsq_5 = reshape(x_bottleneck_Gathe_2, shape1048);

            % Unsqueeze:
            [shape1049, x_bottleneck_UnsqueeNumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_bottleneck_Gather_, this.Vars.onnx__Unsqueeze_276, coder.const(x_bottleneck_Gather_NumDims));
            x_bottleneck_Unsquee = reshape(x_bottleneck_Gather_, shape1049);

            % Concat:
            [x_bottleneck_Conc_6, x_bottleneck_Conc_6NumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {x_bottleneck_Unsq_5, x_bottleneck_Unsquee}, [coder.const(x_bottleneck_Unsq_5NumDims), coder.const(x_bottleneck_UnsqueeNumDims)]);

            % Unsqueeze:
            [shape1050, x_bottleneck_Unsq_1NumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_bottleneck_Gathe_2, this.Vars.onnx__Unsqueeze_279, coder.const(x_bottleneck_Gathe_2NumDims));
            x_bottleneck_Unsq_1 = reshape(x_bottleneck_Gathe_2, shape1050);

            % Unsqueeze:
            [shape1051, x_bottleneck_Unsq_2NumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_bottleneck_Gather_, this.Vars.onnx__Unsqueeze_281, coder.const(x_bottleneck_Gather_NumDims));
            x_bottleneck_Unsq_2 = reshape(x_bottleneck_Gather_, shape1051);

            % Concat:
            [x_bottleneck_Concat_, x_bottleneck_Concat_NumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {x_bottleneck_Unsq_1, x_bottleneck_Unsq_2}, [coder.const(x_bottleneck_Unsq_1NumDims), coder.const(x_bottleneck_Unsq_2NumDims)]);

            % Unsqueeze:
            [shape1052, x_bottleneck_Unsq_3NumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_bottleneck_Gathe_2, this.Vars.onnx__Unsqueeze_284, coder.const(x_bottleneck_Gathe_2NumDims));
            x_bottleneck_Unsq_3 = reshape(x_bottleneck_Gathe_2, shape1052);

            % Unsqueeze:
            [shape1053, x_bottleneck_Unsq_4NumDims] = ege_unet_1024_fp32.coder.ops.prepareUnsqueezeArgs(x_bottleneck_Gather_, this.Vars.onnx__Unsqueeze_286, coder.const(x_bottleneck_Gather_NumDims));
            x_bottleneck_Unsq_4 = reshape(x_bottleneck_Gather_, shape1053);

            % Concat:
            [x_bottleneck_Conc_1, x_bottleneck_Conc_1NumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {x_bottleneck_Unsq_3, x_bottleneck_Unsq_4}, [coder.const(x_bottleneck_Unsq_3NumDims), coder.const(x_bottleneck_Unsq_4NumDims)]);

            % Cast:
            x_bottleneck_Cast_ou = cast(int64(ege_unet_1024_fp32.coder.ops.extractIfDlarray(x_bottleneck_Conc_6)), 'like', x_bottleneck_Conc_6);
            x_bottleneck_Cast_ouNumDims = coder.const(x_bottleneck_Conc_6NumDims);

            % Concat:
            [x_bottleneck_Conc_2, x_bottleneck_Conc_2NumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {this.onnx__Concat_556, x_bottleneck_Cast_ou}, [this.NumDims.onnx__Concat_556, coder.const(x_bottleneck_Cast_ouNumDims)]);

            % Resize:
            [DLTScales1054, DLTSizes1055, dataFormat1056, Method1057, GeometricTransformMode1058, NearestRoundingMode1059, x_bottleneck_Resiz_2NumDims] = ege_unet_1024_fp32.coder.ops.prepareResize11Args(dlarray([]), dlarray([]), x_bottleneck_Conc_2, "half_pixel", "linear", "floor", this.NumDims.onnx__Resize_554);
            X1060 = dlarray(single(ege_unet_1024_fp32.coder.ops.extractIfDlarray(this.onnx__Resize_554)));
            if isempty(DLTScales1054)
                Y1061 = dlresize(X1060, 'OutputSize', DLTSizes1055(1:end-1), 'DataFormat', dataFormat1056, 'Method', Method1057, 'GeometricTransformMode', GeometricTransformMode1058, 'NearestRoundingMode', NearestRoundingMode1059);
            else
                Y1061 = dlresize(X1060, 'Scale', DLTScales1054(1:end-1), 'DataFormat', dataFormat1056, 'Method', Method1057, 'GeometricTransformMode', GeometricTransformMode1058, 'NearestRoundingMode', NearestRoundingMode1059);
            end
            x_bottleneck_Resiz_2 = ege_unet_1024_fp32.coder.ops.extractIfDlarray(Y1061);

            % Mul:
            x_bottleneck_Mul_4_o = x_bottleneck_Slice_o .* x_bottleneck_Resiz_2;
            x_bottleneck_Mul_4_oNumDims = max(coder.const(x_bottleneck_Slice_oNumDims), coder.const(x_bottleneck_Resiz_2NumDims));

            % Cast:
            x_bottleneck_Cast_1_ = cast(int64(ege_unet_1024_fp32.coder.ops.extractIfDlarray(x_bottleneck_Concat_)), 'like', x_bottleneck_Concat_);
            x_bottleneck_Cast_1_NumDims = coder.const(x_bottleneck_Concat_NumDims);

            % Concat:
            [x_bottleneck_Conc_3, x_bottleneck_Conc_3NumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {onnx__Concat_559, x_bottleneck_Cast_1_}, [coder.const(onnx__Concat_559NumDims), coder.const(x_bottleneck_Cast_1_NumDims)]);

            % Resize:
            [DLTScales1062, DLTSizes1063, dataFormat1064, Method1065, GeometricTransformMode1066, NearestRoundingMode1067, x_bottleneck_Resize_NumDims] = ege_unet_1024_fp32.coder.ops.prepareResize11Args(dlarray([]), dlarray([]), x_bottleneck_Conc_3, "half_pixel", "linear", "floor", this.NumDims.onnx__Resize_557);
            X1068 = dlarray(single(ege_unet_1024_fp32.coder.ops.extractIfDlarray(this.onnx__Resize_557)));
            if isempty(DLTScales1062)
                Y1069 = dlresize(X1068, 'OutputSize', DLTSizes1063(1:end-1), 'DataFormat', dataFormat1064, 'Method', Method1065, 'GeometricTransformMode', GeometricTransformMode1066, 'NearestRoundingMode', NearestRoundingMode1067);
            else
                Y1069 = dlresize(X1068, 'Scale', DLTScales1062(1:end-1), 'DataFormat', dataFormat1064, 'Method', Method1065, 'GeometricTransformMode', GeometricTransformMode1066, 'NearestRoundingMode', NearestRoundingMode1067);
            end
            x_bottleneck_Resize_ = ege_unet_1024_fp32.coder.ops.extractIfDlarray(Y1069);

            % Mul:
            x_bottleneck_Mul_5_o = x_bottleneck_Slice_1 .* x_bottleneck_Resize_;
            x_bottleneck_Mul_5_oNumDims = max(coder.const(x_bottleneck_Slice_1NumDims), coder.const(x_bottleneck_Resize_NumDims));

            % Cast:
            x_bottleneck_Cast_2_ = cast(int64(ege_unet_1024_fp32.coder.ops.extractIfDlarray(x_bottleneck_Conc_1)), 'like', x_bottleneck_Conc_1);
            x_bottleneck_Cast_2_NumDims = coder.const(x_bottleneck_Conc_1NumDims);

            % Concat:
            [x_bottleneck_Conc_4, x_bottleneck_Conc_4NumDims] = ege_unet_1024_fp32.coder.ops.onnxConcat(0, {onnx__Concat_562, x_bottleneck_Cast_2_}, [coder.const(onnx__Concat_562NumDims), coder.const(x_bottleneck_Cast_2_NumDims)]);

            % Resize:
            [DLTScales1070, DLTSizes1071, dataFormat1072, Method1073, GeometricTransformMode1074, NearestRoundingMode1075, x_bottleneck_Resiz_1NumDims] = ege_unet_1024_fp32.coder.ops.prepareResize11Args(dlarray([]), dlarray([]), x_bottleneck_Conc_4, "half_pixel", "linear", "floor", this.NumDims.onnx__Resize_560);
            X1076 = dlarray(single(ege_unet_1024_fp32.coder.ops.extractIfDlarray(this.onnx__Resize_560)));
            if isempty(DLTScales1070)
                Y1077 = dlresize(X1076, 'OutputSize', DLTSizes1071(1:end-1), 'DataFormat', dataFormat1072, 'Method', Method1073, 'GeometricTransformMode', GeometricTransformMode1074, 'NearestRoundingMode', NearestRoundingMode1075);
            else
                Y1077 = dlresize(X1076, 'Scale', DLTScales1070(1:end-1), 'DataFormat', dataFormat1072, 'Method', Method1073, 'GeometricTransformMode', GeometricTransformMode1074, 'NearestRoundingMode', NearestRoundingMode1075);
            end
            x_bottleneck_Resiz_1 = ege_unet_1024_fp32.coder.ops.extractIfDlarray(Y1077);

            % Mul:
            x_bottleneck_Mul_6_o = x_bottleneck_Slice_2 .* x_bottleneck_Resiz_1;
            x_bottleneck_Mul_6_oNumDims = max(coder.const(x_bottleneck_Slice_2NumDims), coder.const(x_bottleneck_Resiz_1NumDims));

            % Set graph output arguments
            x_bottleneck_Mul_4_oNumDims1009 = x_bottleneck_Mul_4_oNumDims;
            x_bottleneck_Mul_5_oNumDims1010 = x_bottleneck_Mul_5_oNumDims;
            x_bottleneck_Mul_6_oNumDims1011 = x_bottleneck_Mul_6_oNumDims;
            x_bottleneck_Slice_3NumDims1012 = x_bottleneck_Slice_3NumDims;

        end

    end

end