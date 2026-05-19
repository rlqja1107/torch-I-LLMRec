from transformers import PretrainedConfig


class ILLMRecConfig(PretrainedConfig):
    model_type = "illmrec"

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        data_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        mm_vision_select_layer=None,
        mm_vision_select_feature=None,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=True,
        mm_projector_lr=None,
        vision_resolution=None,
        interpolate_mode=None,
        s2=None,
        s2_scales=None,
        s2_max_split_size=None,
        rec_cfg=None,
        train_cfg=None,
        n_negative=None,
        **kwargs
    ):
        super().__init__()
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.rec_cfg = rec_cfg
        self.train_cfg = train_cfg
        self.resume_path = resume_path
        
        # data argument
        self.n_negative = n_negative
        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_projector_lr = mm_projector_lr
        self.vision_resolution = vision_resolution
        self.interpolate_mode = interpolate_mode
        self.s2 = s2
        self.s2_scales = s2_scales
        self.s2_max_split_size = s2_max_split_size
        self.data_cfg = data_cfg
