# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List
import torch
from torch import nn
from torchvision.ops import RoIPool

from detectron2.layers import ROIAlign, ROIAlignRotated, cat, nonzero_tuple
from detectron2.structures import Boxes

"""
To export ROIPooler to torchscript, in this file, variables that should be annotated with
`Union[List[Boxes], List[RotatedBoxes]]` are only annotated with `List[Boxes]`.

TODO: Correct these annotations when torchscript support `Union`.
https://github.com/pytorch/pytorch/issues/41412
"""

__all__ = ["ROIPooler"]


def assign_boxes_to_levels(
    box_lists: List[Boxes],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full_like(
        box_tensor[:, :1], batch_index, dtype=box_tensor.dtype, device=box_tensor.device
    )
    return cat((repeated_index, box_tensor), dim=1)


def convert_boxes_to_pooler_format(box_lists: List[Boxes]):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """
    pooler_fmt_boxes = cat(
        [_fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
        fixed=True,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size
        self.fixed = fixed

        self.inlayer = nn.Conv2d(256, 256*2, kernel_size=(1,1), padding=0)
        self.reclayer = nn.Conv2d(256, 256*2, kernel_size=(3,3), stride=(2,2), padding=0)
        self.outlayer = nn.Conv2d(256*2, 256, kernel_size=(1,1))
        
        self.var_outlayer = nn.Conv2d(256,256, kernel_size=(2,2), stride=(1,1))

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )
        if len(box_lists) == 0:
            return torch.zeros(
                (0, x[0].shape[1]) + self.output_size, device=x[0].device, dtype=x[0].dtype
            )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = pooler_fmt_boxes.size(0)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros((num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device)

        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]

            boxes = pooler_fmt_boxes[inds]
            scale = pooler.spatial_scale
            boxes[:,1:3] = torch.floor(boxes[:,1:3]*scale)
            boxes[:,3:5] = torch.ceil(boxes[:,3:5]*scale)
            boxes = boxes.to(device=device,dtype=torch.long)

            feats = x[level]

            # replacing the below for loop
            boxes[boxes[:,1]< 0,1] = 0
            boxes[boxes[:,2]< 0,2] = 0
            
            boxes[boxes[:,3] >= feats[0].shape[-1],3] = feats[0].shape[-1]-1
            boxes[boxes[:,4] >= feats[0].shape[-2],4] = feats[0].shape[-2]-1
            
            if boxes.shape[0] < 1:
                continue
            
            height = boxes[:,4] - boxes[:,2] + 1
            width = boxes[:,3] - boxes[:,1] + 1
            max_h,max_w = torch.max(torch.max(height),0)[0], torch.max(torch.max(width),0)[0]
            max_h,max_w = torch.max(torch.tensor([max_h,3])), torch.max(torch.tensor([max_w,3]))
            crops = torch.zeros((boxes.shape[0],256,max_h,max_w),device=device,dtype=torch.float32)
            for i in  range(boxes.shape[0]): 
                ind,x0,y0,x1,y1 = boxes[i]
                crops[i,:,:y1-y0+1,:x1-x0+1] = feats[ind][:,y0:y1+1,x0:x1+1]
            boxes[:,0] = torch.arange(boxes.shape[0]) ## i changes this from 0
            boxes[:,3:5] -= boxes[:,1:3]
            boxes[:,1:3] = 0
            
            if self.fixed:
                crops,boxes = self.fixed_learnable_downsample(crops,boxes, out_shape=self.output_size, device=device)
                output[inds] = pooler(crops,boxes,1.0)
        
        
        return output
    def outShape(self,x,stride,kernel):
        x_out = torch.floor((x-kernel)/(stride*1.0)+1)
        return x_out.type(torch.int32)
    def hwOut(self,hin,win,strides=(1,1),kernel=(3,3)):
        hout = self.outShape(hin,strides[0],kernel[0])
        wout = self.outShape(win,strides[1],kernel[1])
        return hout,wout
    def rcConv(self,x,box_x):
        x1 = self.reclayer(x)
        x1 = self.outlayer(x1)
        #x1 = padding(x1)
        box_x[:,-1],box_x[:,-2] = self.hwOut(box_x[:,-1],box_x[:,-2],strides=(2,2),kernel=(3,3))
        #make box changes
        return x1,box_x
    def fixed_learnable_downsample(self, features,boxes, out_shape=(7,7),kernel_size=(3,3),strides=(2,2),device=None):
        # start edit 3
        #mask_omit = torch.ones((boxes.shape[0]),dtype=torch.bool,device=device)
        result_x = torch.zeros(features.size(),dtype=torch.float,device=device)
        result_box = torch.zeros(boxes.size(),device=device,dtype=torch.long)
        #N,c,m,n = features.shape
        features_ = features
        boxes_ = boxes
        indexes = torch.arange(boxes.shape[0])
        while indexes.shape[0]>0:
            mask = torch.logical_and(
                self.outShape(boxes_[:,-1],strides[0],kernel_size[0]) > out_shape[0],
                self.outShape(boxes_[:,-2],strides[1],kernel_size[1]) > out_shape[1]
            )
            mask_not = torch.logical_not(mask)
            indexes_ = indexes[mask]
            finalized =  indexes[mask_not]
            # port finalized to results
            result_x[finalized,:,:features_.shape[2],:features_.shape[3]] = features_[mask_not,:,:,:]
            result_box[finalized,:] = boxes_[mask_not,:]
            features_,boxes_ = self.rcConv(features_[mask],boxes_[mask])
            indexes = indexes_
        return result_x,result_box
