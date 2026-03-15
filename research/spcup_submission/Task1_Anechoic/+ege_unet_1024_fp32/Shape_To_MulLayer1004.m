classdef Shape_To_MulLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

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




    methods
        function this = Shape_To_MulLayer1004(name)
            this.Name = name;
            this.NumOutputs = 4;
            this.OutputNames = {'x_bottleneck_Mul_4_o', 'x_bottleneck_Mul_5_o', 'x_bottleneck_Mul_6_o', 'x_bottleneck_Slice_3'};
        end

        function [x_bottleneck_Mul_4_o, x_bottleneck_Mul_5_o, x_bottleneck_Mul_6_o, x_bottleneck_Slice_3] = predict(this, x_pool_3_MaxPool_out)
            if isdlarray(x_pool_3_MaxPool_out)
                x_pool_3_MaxPool_out = stripdims(x_pool_3_MaxPool_out);
            end
            x_pool_3_MaxPool_outNumDims = 4;
            x_pool_3_MaxPool_out = ege_unet_1024_fp32.ops.permuteInputVar(x_pool_3_MaxPool_out, [4 3 1 2], 4);

            [x_bottleneck_Mul_4_o, x_bottleneck_Mul_5_o, x_bottleneck_Mul_6_o, x_bottleneck_Slice_3, x_bottleneck_Mul_4_oNumDims, x_bottleneck_Mul_5_oNumDims, x_bottleneck_Mul_6_oNumDims, x_bottleneck_Slice_3NumDims] = Shape_To_MulGraph1008(this, x_pool_3_MaxPool_out, x_pool_3_MaxPool_outNumDims, false);
            x_bottleneck_Mul_4_o = ege_unet_1024_fp32.ops.permuteOutputVar(x_bottleneck_Mul_4_o, [3 4 2 1], 4);
            x_bottleneck_Mul_5_o = ege_unet_1024_fp32.ops.permuteOutputVar(x_bottleneck_Mul_5_o, [3 4 2 1], 4);
            x_bottleneck_Mul_6_o = ege_unet_1024_fp32.ops.permuteOutputVar(x_bottleneck_Mul_6_o, [3 4 2 1], 4);
            x_bottleneck_Slice_3 = ege_unet_1024_fp32.ops.permuteOutputVar(x_bottleneck_Slice_3, [3 4 2 1], 4);

            x_bottleneck_Mul_4_o = dlarray(single(x_bottleneck_Mul_4_o), 'SSCB');
            x_bottleneck_Mul_5_o = dlarray(single(x_bottleneck_Mul_5_o), 'SSCB');
            x_bottleneck_Mul_6_o = dlarray(single(x_bottleneck_Mul_6_o), 'SSCB');
            x_bottleneck_Slice_3 = dlarray(single(x_bottleneck_Slice_3), 'SSCB');
        end

        function [x_bottleneck_Mul_4_o, x_bottleneck_Mul_5_o, x_bottleneck_Mul_6_o, x_bottleneck_Slice_3] = forward(this, x_pool_3_MaxPool_out)
            if isdlarray(x_pool_3_MaxPool_out)
                x_pool_3_MaxPool_out = stripdims(x_pool_3_MaxPool_out);
            end
            x_pool_3_MaxPool_outNumDims = 4;
            x_pool_3_MaxPool_out = ege_unet_1024_fp32.ops.permuteInputVar(x_pool_3_MaxPool_out, [4 3 1 2], 4);

            [x_bottleneck_Mul_4_o, x_bottleneck_Mul_5_o, x_bottleneck_Mul_6_o, x_bottleneck_Slice_3, x_bottleneck_Mul_4_oNumDims, x_bottleneck_Mul_5_oNumDims, x_bottleneck_Mul_6_oNumDims, x_bottleneck_Slice_3NumDims] = Shape_To_MulGraph1008(this, x_pool_3_MaxPool_out, x_pool_3_MaxPool_outNumDims, true);
            x_bottleneck_Mul_4_o = ege_unet_1024_fp32.ops.permuteOutputVar(x_bottleneck_Mul_4_o, [3 4 2 1], 4);
            x_bottleneck_Mul_5_o = ege_unet_1024_fp32.ops.permuteOutputVar(x_bottleneck_Mul_5_o, [3 4 2 1], 4);
            x_bottleneck_Mul_6_o = ege_unet_1024_fp32.ops.permuteOutputVar(x_bottleneck_Mul_6_o, [3 4 2 1], 4);
            x_bottleneck_Slice_3 = ege_unet_1024_fp32.ops.permuteOutputVar(x_bottleneck_Slice_3, [3 4 2 1], 4);

            x_bottleneck_Mul_4_o = dlarray(single(x_bottleneck_Mul_4_o), 'SSCB');
            x_bottleneck_Mul_5_o = dlarray(single(x_bottleneck_Mul_5_o), 'SSCB');
            x_bottleneck_Mul_6_o = dlarray(single(x_bottleneck_Mul_6_o), 'SSCB');
            x_bottleneck_Slice_3 = dlarray(single(x_bottleneck_Slice_3), 'SSCB');
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
            [x_bottleneck_Shape_o, x_bottleneck_Shape_oNumDims] = ege_unet_1024_fp32.ops.onnxShape(x_pool_3_MaxPool_out, x_pool_3_MaxPool_outNumDims, 0, x_pool_3_MaxPool_outNumDims+1);

            % Gather:
            [x_bottleneck_Gathe_2, x_bottleneck_Gathe_2NumDims] = ege_unet_1024_fp32.ops.onnxGather(x_bottleneck_Shape_o, this.Vars.x_bottleneck_Cons_9, 0, x_bottleneck_Shape_oNumDims, this.NumDims.x_bottleneck_Cons_9);

            % Shape:
            [x_bottleneck_Shape_1, x_bottleneck_Shape_1NumDims] = ege_unet_1024_fp32.ops.onnxShape(x_pool_3_MaxPool_out, x_pool_3_MaxPool_outNumDims, 0, x_pool_3_MaxPool_outNumDims+1);

            % Gather:
            [x_bottleneck_Gather_, x_bottleneck_Gather_NumDims] = ege_unet_1024_fp32.ops.onnxGather(x_bottleneck_Shape_1, this.Vars.x_bottleneck_Constan, 0, x_bottleneck_Shape_1NumDims, this.NumDims.x_bottleneck_Constan);

            % Shape:
            [x_bottleneck_Shape_2, x_bottleneck_Shape_2NumDims] = ege_unet_1024_fp32.ops.onnxShape(x_pool_3_MaxPool_out, x_pool_3_MaxPool_outNumDims, 0, x_pool_3_MaxPool_outNumDims+1);

            % Gather:
            [x_bottleneck_Gathe_1, x_bottleneck_Gathe_1NumDims] = ege_unet_1024_fp32.ops.onnxGather(x_bottleneck_Shape_2, this.Vars.x_bottleneck_Cons_1, 0, x_bottleneck_Shape_2NumDims, this.NumDims.x_bottleneck_Cons_1);

            % Add:
            x_bottleneck_Add_out = x_bottleneck_Gathe_1 + this.Vars.x_bottleneck_Cons_3;
            x_bottleneck_Add_outNumDims = max(x_bottleneck_Gathe_1NumDims, this.NumDims.x_bottleneck_Cons_3);

            % Div:
            x_bottleneck_Div_out = fix(x_bottleneck_Add_out ./ this.Vars.x_bottleneck_Cons_4);
            x_bottleneck_Div_outNumDims = max(x_bottleneck_Add_outNumDims, this.NumDims.x_bottleneck_Cons_4);

            % Mul:
            x_bottleneck_Mul_out = x_bottleneck_Div_out .* this.Vars.x_bottleneck_Cons_5;
            x_bottleneck_Mul_outNumDims = max(x_bottleneck_Div_outNumDims, this.NumDims.x_bottleneck_Cons_5);

            % Slice:
            [Indices, x_bottleneck_Slice_oNumDims] = ege_unet_1024_fp32.ops.prepareSliceArgs(x_pool_3_MaxPool_out, this.Vars.x_bottleneck_Cons_2, x_bottleneck_Mul_out, this.Vars.x_bottleneck_Cons_1, '', x_pool_3_MaxPool_outNumDims);
            x_bottleneck_Slice_o = x_pool_3_MaxPool_out(Indices{:});

            % Mul:
            x_bottleneck_Mul_1_o = x_bottleneck_Div_out .* this.Vars.x_bottleneck_Cons_6;
            x_bottleneck_Mul_1_oNumDims = max(x_bottleneck_Div_outNumDims, this.NumDims.x_bottleneck_Cons_6);

            % Slice:
            [Indices, x_bottleneck_Slice_1NumDims] = ege_unet_1024_fp32.ops.prepareSliceArgs(x_pool_3_MaxPool_out, x_bottleneck_Mul_out, x_bottleneck_Mul_1_o, this.Vars.x_bottleneck_Cons_1, '', x_pool_3_MaxPool_outNumDims);
            x_bottleneck_Slice_1 = x_pool_3_MaxPool_out(Indices{:});

            % Mul:
            x_bottleneck_Mul_2_o = x_bottleneck_Div_out .* this.Vars.x_bottleneck_Cons_7;
            x_bottleneck_Mul_2_oNumDims = max(x_bottleneck_Div_outNumDims, this.NumDims.x_bottleneck_Cons_7);

            % Slice:
            [Indices, x_bottleneck_Slice_2NumDims] = ege_unet_1024_fp32.ops.prepareSliceArgs(x_pool_3_MaxPool_out, x_bottleneck_Mul_1_o, x_bottleneck_Mul_2_o, this.Vars.x_bottleneck_Cons_1, '', x_pool_3_MaxPool_outNumDims);
            x_bottleneck_Slice_2 = x_pool_3_MaxPool_out(Indices{:});

            % Mul:
            x_bottleneck_Mul_3_o = x_bottleneck_Div_out .* this.Vars.x_bottleneck_Cons_8;
            x_bottleneck_Mul_3_oNumDims = max(x_bottleneck_Div_outNumDims, this.NumDims.x_bottleneck_Cons_8);

            % Slice:
            [Indices, x_bottleneck_Slice_3NumDims] = ege_unet_1024_fp32.ops.prepareSliceArgs(x_pool_3_MaxPool_out, x_bottleneck_Mul_2_o, x_bottleneck_Mul_3_o, this.Vars.x_bottleneck_Cons_1, '', x_pool_3_MaxPool_outNumDims);
            x_bottleneck_Slice_3 = x_pool_3_MaxPool_out(Indices{:});

            % Unsqueeze:
            [shape, x_bottleneck_Unsq_5NumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_bottleneck_Gathe_2, this.Vars.onnx__Unsqueeze_274, x_bottleneck_Gathe_2NumDims);
            x_bottleneck_Unsq_5 = reshape(x_bottleneck_Gathe_2, shape);

            % Unsqueeze:
            [shape, x_bottleneck_UnsqueeNumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_bottleneck_Gather_, this.Vars.onnx__Unsqueeze_276, x_bottleneck_Gather_NumDims);
            x_bottleneck_Unsquee = reshape(x_bottleneck_Gather_, shape);

            % Concat:
            [x_bottleneck_Conc_6, x_bottleneck_Conc_6NumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {x_bottleneck_Unsq_5, x_bottleneck_Unsquee}, [x_bottleneck_Unsq_5NumDims, x_bottleneck_UnsqueeNumDims]);

            % Unsqueeze:
            [shape, x_bottleneck_Unsq_1NumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_bottleneck_Gathe_2, this.Vars.onnx__Unsqueeze_279, x_bottleneck_Gathe_2NumDims);
            x_bottleneck_Unsq_1 = reshape(x_bottleneck_Gathe_2, shape);

            % Unsqueeze:
            [shape, x_bottleneck_Unsq_2NumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_bottleneck_Gather_, this.Vars.onnx__Unsqueeze_281, x_bottleneck_Gather_NumDims);
            x_bottleneck_Unsq_2 = reshape(x_bottleneck_Gather_, shape);

            % Concat:
            [x_bottleneck_Concat_, x_bottleneck_Concat_NumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {x_bottleneck_Unsq_1, x_bottleneck_Unsq_2}, [x_bottleneck_Unsq_1NumDims, x_bottleneck_Unsq_2NumDims]);

            % Unsqueeze:
            [shape, x_bottleneck_Unsq_3NumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_bottleneck_Gathe_2, this.Vars.onnx__Unsqueeze_284, x_bottleneck_Gathe_2NumDims);
            x_bottleneck_Unsq_3 = reshape(x_bottleneck_Gathe_2, shape);

            % Unsqueeze:
            [shape, x_bottleneck_Unsq_4NumDims] = ege_unet_1024_fp32.ops.prepareUnsqueezeArgs(x_bottleneck_Gather_, this.Vars.onnx__Unsqueeze_286, x_bottleneck_Gather_NumDims);
            x_bottleneck_Unsq_4 = reshape(x_bottleneck_Gather_, shape);

            % Concat:
            [x_bottleneck_Conc_1, x_bottleneck_Conc_1NumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {x_bottleneck_Unsq_3, x_bottleneck_Unsq_4}, [x_bottleneck_Unsq_3NumDims, x_bottleneck_Unsq_4NumDims]);

            % Cast:
            x_bottleneck_Cast_ou = cast(int64(extractdata(x_bottleneck_Conc_6)), 'like', x_bottleneck_Conc_6);
            x_bottleneck_Cast_ouNumDims = x_bottleneck_Conc_6NumDims;

            % Concat:
            [x_bottleneck_Conc_2, x_bottleneck_Conc_2NumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {this.onnx__Concat_556, x_bottleneck_Cast_ou}, [this.NumDims.onnx__Concat_556, x_bottleneck_Cast_ouNumDims]);

            % Resize:
            [DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, x_bottleneck_Resiz_2NumDims] = ege_unet_1024_fp32.ops.prepareResize11Args(dlarray([]), dlarray([]), x_bottleneck_Conc_2, "half_pixel", "linear", "floor", this.NumDims.onnx__Resize_554);
            if isempty(DLTScales)
                x_bottleneck_Resiz_2 = dlresize(this.onnx__Resize_554, 'OutputSize', DLTSizes, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            else
                x_bottleneck_Resiz_2 = dlresize(this.onnx__Resize_554, 'Scale', DLTScales, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            end

            % Mul:
            x_bottleneck_Mul_4_o = x_bottleneck_Slice_o .* x_bottleneck_Resiz_2;
            x_bottleneck_Mul_4_oNumDims = max(x_bottleneck_Slice_oNumDims, x_bottleneck_Resiz_2NumDims);

            % Cast:
            x_bottleneck_Cast_1_ = cast(int64(extractdata(x_bottleneck_Concat_)), 'like', x_bottleneck_Concat_);
            x_bottleneck_Cast_1_NumDims = x_bottleneck_Concat_NumDims;

            % Concat:
            [x_bottleneck_Conc_3, x_bottleneck_Conc_3NumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {onnx__Concat_559, x_bottleneck_Cast_1_}, [onnx__Concat_559NumDims, x_bottleneck_Cast_1_NumDims]);

            % Resize:
            [DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, x_bottleneck_Resize_NumDims] = ege_unet_1024_fp32.ops.prepareResize11Args(dlarray([]), dlarray([]), x_bottleneck_Conc_3, "half_pixel", "linear", "floor", this.NumDims.onnx__Resize_557);
            if isempty(DLTScales)
                x_bottleneck_Resize_ = dlresize(this.onnx__Resize_557, 'OutputSize', DLTSizes, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            else
                x_bottleneck_Resize_ = dlresize(this.onnx__Resize_557, 'Scale', DLTScales, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            end

            % Mul:
            x_bottleneck_Mul_5_o = x_bottleneck_Slice_1 .* x_bottleneck_Resize_;
            x_bottleneck_Mul_5_oNumDims = max(x_bottleneck_Slice_1NumDims, x_bottleneck_Resize_NumDims);

            % Cast:
            x_bottleneck_Cast_2_ = cast(int64(extractdata(x_bottleneck_Conc_1)), 'like', x_bottleneck_Conc_1);
            x_bottleneck_Cast_2_NumDims = x_bottleneck_Conc_1NumDims;

            % Concat:
            [x_bottleneck_Conc_4, x_bottleneck_Conc_4NumDims] = ege_unet_1024_fp32.ops.onnxConcat(0, {onnx__Concat_562, x_bottleneck_Cast_2_}, [onnx__Concat_562NumDims, x_bottleneck_Cast_2_NumDims]);

            % Resize:
            [DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, x_bottleneck_Resiz_1NumDims] = ege_unet_1024_fp32.ops.prepareResize11Args(dlarray([]), dlarray([]), x_bottleneck_Conc_4, "half_pixel", "linear", "floor", this.NumDims.onnx__Resize_560);
            if isempty(DLTScales)
                x_bottleneck_Resiz_1 = dlresize(this.onnx__Resize_560, 'OutputSize', DLTSizes, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            else
                x_bottleneck_Resiz_1 = dlresize(this.onnx__Resize_560, 'Scale', DLTScales, 'DataFormat', dataFormat, 'Method', Method, 'GeometricTransformMode', GeometricTransformMode, 'NearestRoundingMode', NearestRoundingMode);
            end

            % Mul:
            x_bottleneck_Mul_6_o = x_bottleneck_Slice_2 .* x_bottleneck_Resiz_1;
            x_bottleneck_Mul_6_oNumDims = max(x_bottleneck_Slice_2NumDims, x_bottleneck_Resiz_1NumDims);

            % Set graph output arguments
            x_bottleneck_Mul_4_oNumDims1009 = x_bottleneck_Mul_4_oNumDims;
            x_bottleneck_Mul_5_oNumDims1010 = x_bottleneck_Mul_5_oNumDims;
            x_bottleneck_Mul_6_oNumDims1011 = x_bottleneck_Mul_6_oNumDims;
            x_bottleneck_Slice_3NumDims1012 = x_bottleneck_Slice_3NumDims;

        end

    end

end