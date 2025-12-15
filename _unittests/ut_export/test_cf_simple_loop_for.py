import unittest
from typing import Tuple
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_torch
from onnx_diagnostic.export.control_flow_onnx import (
    enable_code_export_control_flow,
)
from onnx_diagnostic.export.cf_simple_loop_for import simple_loop_for, SimpleLoopForOp
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.export.api import to_onnx


class TestCfSimpleLoopFor(ExtTestCase):
    @requires_torch("2.9.99")
    def test_simple_loop_for_int(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1),)

                return simple_loop_for(4, body, (x,))

        model = Model()
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1)
        got = model(x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (x,), dynamic_shapes=(({0: torch.export.Dim.DYNAMIC},))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        # Loop should be unrolled.
        self.assertEqual(len(check), 0)
        got = ep.module()(x)
        self.assertEqualArray(expected, got)

    @requires_torch("2.9.99")
    def test_simple_loop_for_no_inputs(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (torch.arange(i + 1, dtype=torch.int64),)

                y = simple_loop_for(n_iter, body)
                torch._check(isinstance(y, torch.Tensor), lambda: f"y is {type(y)}")
                return x.unsqueeze(1) + y.unsqueeze(0).to(x.device)

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(8, dtype=torch.float32)
        expected = x.reshape((-1, 1)) + torch.tensor(
            [[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]], dtype=x.dtype
        )
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_1(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1),)

                return simple_loop_for(n_iter, body, (x,))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_1_module(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1),)

                return simple_loop_for(n_iter, body, (x,))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        got = ep.module()(n_iter, x)
        self.assertEqualArray(expected, got)

    @requires_torch("2.9.99")
    def test_simple_loop_for_2(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(
                    i: torch.Tensor, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1), x[i.item() + 1 :].unsqueeze(1))

                return simple_loop_for(n_iter, body, (x,))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = (
            torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1),
            torch.tensor(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                ],
                dtype=x.dtype,
            ).unsqueeze(1),
        )
        got = model(n_iter, x)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_2_concatenation_dims(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(
                    i: torch.Tensor, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    return (
                        x[: i.item() + 1].unsqueeze(1),
                        x[i.item() + 1 :].unsqueeze(0),
                    )

                return simple_loop_for(n_iter, body, (x,), (0, 1))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = (
            torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1),
            torch.tensor(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                ],
                dtype=x.dtype,
            ).unsqueeze(0),
        )
        got = model(n_iter, x)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_1_with_concatenation_dims(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(0),)

                return simple_loop_for(n_iter, body, (x,), 1)

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(0)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_phi4(self):
        _IMAGE_SPECIAL_TOKEN_ID = 200010
        vocab_size = 200064
        hidden_size = 3072
        padding_idx = 199999
        num_img_tokens = 256
        image_dim_out = 1152
        crop_size = 448

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size, padding_idx)
                self.img_projection = torch.nn.Linear(image_dim_out, hidden_size)

            def forward(
                self,
                input_ids,
                hidden_states,
                img_features,
                image_attention_mask,
                img_embeds,
                img_sizes,
            ):
                base_feat_height_reduction = 1

                glb_GN = torch.zeros([1, 1, image_dim_out * base_feat_height_reduction**2])
                sub_GN = torch.zeros([1, 1, 1, image_dim_out * base_feat_height_reduction**2])

                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])

                # positions = torch.nonzero(
                #   input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)
                positions_tuple = torch.nonzero(
                    input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True
                )

                # base_feat_height_target = self.base_feat_height_target
                base_resolution = crop_size
                base_feat_height_reduction = base_feat_height_reduction

                base_feat_height = base_feat_width = torch.sym_int(
                    img_features.shape[1] ** 0.5
                )

                # bs x max_num_crops x (24x24) x C
                bs = img_embeds.shape[0]
                img_features = img_features.view(
                    bs, -1, base_feat_height * base_feat_width, image_dim_out
                )
                C = image_dim_out
                H = base_feat_height

                output_imgs = []
                output_len = []
                # training is tensor, inference is list
                if isinstance(img_sizes, torch.Tensor):
                    img_sizes = img_sizes.view(-1, 2)
                for _bs in range(bs):
                    h, w = img_sizes[_bs][0].item(), img_sizes[_bs][1].item()
                    h = h // base_resolution
                    w = w // base_resolution
                    B_ = h * w

                    # 1 x (24x24) x 1024
                    global_img_feature = img_features[_bs, :1]

                    # 1 x 12 x 12 x 4096
                    glb_img = (
                        global_img_feature.reshape(1, H, H, C)
                        .reshape(
                            1,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            C,
                        )
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(
                            1,
                            H // base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        )
                        .contiguous()
                    )
                    temp_glb_GN = sub_GN.repeat(1, H // base_feat_height_reduction, 1, 1)

                    # 1 x 156 x 4096
                    glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(
                        1, -1, base_feat_height_reduction * base_feat_height_reduction * C
                    )

                    # (max_num_crops-1) x (12x12) x C
                    sub_img = img_features[_bs, 1:]
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[:B_]

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024)
                    # -> (num_crops, 12*12, 4*1024)
                    sub_img = (
                        sub_img.reshape(B_, H, H, C)
                        .reshape(
                            B_,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            C,
                        )
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(
                            B_, -1, base_feat_height_reduction * base_feat_height_reduction * C
                        )
                        .contiguous()
                    )
                    sub_img = (
                        sub_img.reshape(
                            1,
                            h,
                            w,
                            base_feat_height // base_feat_height_reduction,
                            base_feat_width // base_feat_height_reduction,
                            -1,
                        )
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(
                            1,
                            h * base_feat_height // base_feat_height_reduction,
                            w * base_feat_width // base_feat_height_reduction,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        )
                    )

                    if image_attention_mask is not None and len(image_attention_mask) > 0:
                        reshaped_image_attention_mask = (
                            image_attention_mask[_bs, 1 : B_ + 1, 0::2, 0::2]
                            .reshape(
                                1,
                                h,
                                w,
                                base_feat_height // base_feat_height_reduction,
                                base_feat_width // base_feat_height_reduction,
                            )
                            .permute(0, 1, 3, 2, 4)
                            .reshape(
                                1,
                                h * base_feat_height // base_feat_height_reduction,
                                w * base_feat_width // base_feat_height_reduction,
                            )
                        )
                        useful_height = (
                            reshaped_image_attention_mask[0, :, 0].sum().to(torch.int64).item()
                        )
                        useful_width = (
                            reshaped_image_attention_mask[0, 0, :].sum().to(torch.int64).item()
                        )
                        sub_img = sub_img[:, :useful_height, :useful_width]
                        temp_sub_GN = sub_GN.repeat(1, useful_height, 1, 1)
                        temp_len = (
                            image_attention_mask[_bs, : B_ + 1, 0::2, 0::2]
                            .sum()
                            .to(torch.int64)
                            .item()
                            + (useful_height + 1)
                            + base_feat_height // base_feat_height_reduction
                        )
                    else:
                        temp_sub_GN = sub_GN.repeat(
                            1, h * base_feat_height // base_feat_height_reduction, 1, 1
                        )
                        temp_len = int(
                            (h * w + 1) * num_img_tokens
                            + 1
                            + (h + 1) * base_feat_height // base_feat_height_reduction
                        )

                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(
                        1, -1, base_feat_height_reduction * base_feat_height_reduction * C
                    )
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    # glb_sub
                    # output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    # sub_glb
                    output_imgs.append(torch.cat([sub_img, glb_GN, glb_img], dim=1))
                    output_len.append(temp_len)

                img_set_tensor = []
                for _output_img in output_imgs:
                    img_feature_proj = self.img_projection(_output_img)
                    img_set_tensor.append(img_feature_proj)

                # Shape: (merged_N_tokens, C)
                merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
                merged_img_set_tensor = merged_img_set_tensor.to(hidden_states.dtype).to(
                    hidden_states.device
                )
                # Temporarily disable autocast to avoid issue on bf16 tensors
                # Ref: https://github.com/pytorch/pytorch/issues/132715
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    merged_img_set_tensor = merged_img_set_tensor[
                        : positions_tuple[0].shape[0]
                    ]
                    new_hidden_states = hidden_states.index_put(
                        indices=positions_tuple, values=merged_img_set_tensor, accumulate=False
                    )
                hidden_states = new_hidden_states
                return hidden_states

        def body_fn(
            _bs,
            img_features,
            img_sizes,
            image_attention_mask,
            cst_shape_CH,
            glb_GN,
            sub_GN,
            base_resolution=None,
            base_feat_height_reduction=None,
            base_feat_height=None,
            base_feat_width=None,
            self_img_projection=None,
        ):
            # oddly, it seems impossible to write img_sizes[_bs.item()]
            # it needs img_sizes[_bs.item() : (_bs + 1).item()][0]
            row = img_sizes[_bs.item() : (_bs + 1).item()]
            row = row[0]
            h, w = row[0], row[1]
            h = h // base_resolution
            w = w // base_resolution
            B_ = h * w
            C, H = cst_shape_CH.shape

            # 1 x (24x24) x 1024
            global_img_feature = img_features[_bs.item() : (_bs + 1).item(), :1][0]

            # 1 x 12 x 12 x 4096
            glb_img = (
                global_img_feature.reshape(1, H, H, C)
                .reshape(
                    1,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction,
                    C,
                )
                .contiguous()
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(
                    1,
                    H // base_feat_height_reduction,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction * base_feat_height_reduction * C,
                )
                .contiguous()
            )
            temp_glb_GN = sub_GN.repeat(1, H // base_feat_height_reduction, 1, 1)

            # 1 x 156 x 4096
            glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(
                1, -1, base_feat_height_reduction * base_feat_height_reduction * C
            )

            # (max_num_crops-1) x (12x12) x C
            sub_img = img_features[_bs.item() : (_bs + 1).item(), 1:][0]
            # 16x574x1024
            # get rid of padding sub_img
            sub_img = sub_img[: B_.item()]

            # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024)
            # -> (num_crops, 12*12, 4*1024)
            sub_img = (
                sub_img.reshape(B_.item(), H, H, C)
                .reshape(
                    B_.item(),
                    H // base_feat_height_reduction,
                    base_feat_height_reduction,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction,
                    C,
                )
                .contiguous()
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(
                    B_.item(),
                    -1,
                    base_feat_height_reduction * base_feat_height_reduction * C,
                )
                .contiguous()
            )
            sub_img = (
                sub_img.reshape(
                    1,
                    h.item(),
                    w.item(),
                    base_feat_height // base_feat_height_reduction,
                    base_feat_width // base_feat_height_reduction,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(
                    1,
                    (h * base_feat_height // base_feat_height_reduction).item(),
                    (w * base_feat_width // base_feat_height_reduction).item(),
                    base_feat_height_reduction * base_feat_height_reduction * C,
                )
            )

            reshaped_image_attention_mask = (
                image_attention_mask[
                    _bs.item() : (_bs + 1).item(), 1 : (B_ + 1).item(), 0::2, 0::2
                ][0]
                .reshape(
                    1,
                    h.item(),
                    w.item(),
                    base_feat_height // base_feat_height_reduction,
                    base_feat_width // base_feat_height_reduction,
                )
                .permute(0, 1, 3, 2, 4)
                .reshape(
                    1,
                    (h * base_feat_height // base_feat_height_reduction).item(),
                    (w * base_feat_width // base_feat_height_reduction).item(),
                )
            )
            useful_height = reshaped_image_attention_mask[0, :, 0].sum().to(torch.int64).item()
            useful_width = reshaped_image_attention_mask[0, 0, :].sum().to(torch.int64).item()
            # the module cannot be extracted from here
            sub_img = sub_img[:, :useful_height, :useful_width]
            temp_sub_GN = sub_GN.repeat(1, useful_height, 1, 1)
            # temp_len = (
            #    image_attention_mask[_bs, : B_ + 1, 0::2, 0::2]
            #    .sum()
            #    .to(torch.int64)
            #    .item()
            #    + (useful_height + 1)
            #    + base_feat_height // base_feat_height_reduction
            # )

            sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(
                1, -1, base_feat_height_reduction * base_feat_height_reduction * C
            )
            # (1, num_img_tokens, 1024*4)

            # glb + sub
            # glb_sub
            # output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
            # sub_glb
            _output_img = torch.cat([sub_img, glb_GN, glb_img], dim=1)
            # output_len.append(temp_len)
            proj = self_img_projection(_output_img)
            return (proj,)

        class RewrittenModelLoop(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.embed_tokens = model.embed_tokens
                self.img_projection = model.img_projection

            def forward(
                self,
                input_ids,
                hidden_states,
                img_features,
                image_attention_mask,
                img_embeds,
                img_sizes,
            ):
                base_feat_height_reduction = 1

                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])

                # positions = torch.nonzero(
                #   input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)
                positions_tuple = torch.nonzero(
                    input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True
                )

                # base_feat_height_target = self.base_feat_height_target
                base_resolution = crop_size
                base_feat_height_reduction = base_feat_height_reduction

                base_feat_height = base_feat_width = torch.sym_int(
                    img_features.shape[1] ** 0.5
                )

                # bs x max_num_crops x (24x24) x C
                bs = img_embeds.shape[0]
                img_features = img_features.view(
                    bs, -1, base_feat_height * base_feat_width, image_dim_out
                )
                C = image_dim_out
                H = base_feat_height
                cst_shape_CH = torch.zeros((C, H), dtype=torch.int32)

                # training is tensor, inference is list
                if isinstance(img_sizes, torch.Tensor):
                    img_sizes = img_sizes.view(-1, 2)

                def local_body_fn(
                    n_iter,
                    img_features,
                    img_sizes,
                    image_attention_mask,
                    cst_shape_CH,
                    glb_GN,
                    sub_GN,
                ):
                    return body_fn(
                        n_iter,
                        img_features,
                        img_sizes,
                        image_attention_mask,
                        cst_shape_CH,
                        glb_GN,
                        sub_GN,
                        base_resolution=base_resolution,
                        base_feat_height_reduction=base_feat_height_reduction,
                        base_feat_height=base_feat_height,
                        base_feat_width=base_feat_width,
                        self_img_projection=self.img_projection,
                    )

                tmp = torch.arange(bs + 1).max()
                glb_GN = torch.zeros([1, 1, image_dim_out * base_feat_height_reduction**2])
                sub_GN = torch.zeros([1, 1, 1, image_dim_out * base_feat_height_reduction**2])
                merged_img_set_tensor = simple_loop_for(
                    tmp,
                    local_body_fn,
                    (
                        img_features,
                        img_sizes,
                        image_attention_mask,
                        cst_shape_CH,
                        glb_GN,
                        sub_GN,
                    ),
                    [1],
                )
                merged_img_set_tensor = merged_img_set_tensor.squeeze(0)
                merged_img_set_tensor = merged_img_set_tensor.to(hidden_states.dtype).to(
                    hidden_states.device
                )
                # Temporarily disable autocast to avoid issue on bf16 tensors
                # Ref: https://github.com/pytorch/pytorch/issues/132715
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    merged_img_set_tensor = merged_img_set_tensor[
                        : positions_tuple[0].shape[0]
                    ]
                    new_hidden_states = hidden_states.index_put(
                        indices=positions_tuple, values=merged_img_set_tensor, accumulate=False
                    )
                hidden_states = new_hidden_states
                return hidden_states

        model = Model()
        model.eval()
        input_ids = torch.randint(0, 180000, (2, 9246), dtype=torch.int64)
        input_ids[0, :1000] = _IMAGE_SPECIAL_TOKEN_ID
        hidden_states = model.embed_tokens(input_ids)
        img_features = torch.rand((116, 256, 1152), dtype=torch.float16)
        image_attention_mask = torch.rand((2, 29, 32, 32), dtype=torch.float16)
        img_embeds = torch.rand((2, 29, 3, 448, 448), dtype=torch.float16)
        img_sizes = torch.tensor([[896, 1344], [1792, 3136]], dtype=torch.int64)
        expected = model(
            input_ids, hidden_states, img_features, image_attention_mask, img_embeds, img_sizes
        )
        self.assertEqual(expected.shape, (2, 9246, 3072))

        rewritten_model = RewrittenModelLoop(model)
        rewritten_model.eval()
        patched = rewritten_model(
            input_ids,
            hidden_states,
            img_features,
            image_attention_mask,
            img_embeds,
            img_sizes,
        )
        self.assertEqualArray(expected, patched)
        dynamic_shapes = (
            {0: "batch_size", 1: "seq_length"},
            {0: "A"},
            {0: "B", 1: "C"},
            {0: "batch_size", 1: "new_length"},
            {0: "batch_size", 1: "new_lenght"},
            {0: "batch_size"},
        )
        ep = torch.export.export(
            rewritten_model,
            (
                input_ids,
                hidden_states,
                img_features,
                image_attention_mask,
                img_embeds,
                img_sizes,
            ),
            dynamic_shapes=use_dyn_not_str(dynamic_shapes),
        )

        raise unittest.SkipTest("how to deal with a body calling a submodule?")
        onx = to_onnx(
            rewritten_model,
            (
                input_ids,
                hidden_states,
                img_features,
                image_attention_mask,
                img_embeds,
                img_sizes,
            ),
            dynamic_shapes=dynamic_shapes,
            exporter="custom",
        )
        self.assertNotEmpty(onx)

        # does not work
        # ep = ep.run_decompositions()
        # print(ep)
        # does not work either
        # This part does not work.
        got = ep.module()(
            input_ids,
            hidden_states,
            img_features,
            image_attention_mask,
            img_embeds,
            img_sizes,
        )
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
