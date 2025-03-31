import io
import functools
import textwrap
from typing import Dict, List

__date__ = "2025-03-26"

__data_arch__ = textwrap.dedent(
    """
    architecture,task
    ASTModel,feature-extraction
    AlbertModel,feature-extraction
    BeitForImageClassification,image-classification
    BigBirdModel,feature-extraction
    BlenderbotModel,feature-extraction
    BloomModel,feature-extraction
    CLIPModel,zero-shot-image-classification
    CLIPVisionModel,feature-extraction
    CamembertModel,feature-extraction
    CodeGenModel,feature-extraction
    ConvBertModel,feature-extraction
    ConvNextForImageClassification,image-classification
    ConvNextV2Model,image-feature-extraction
    CvtModel,feature-extraction
    DPTModel,image-feature-extraction
    Data2VecAudioModel,feature-extraction
    Data2VecTextModel,feature-extraction
    Data2VecVisionModel,image-feature-extraction
    DebertaModel,feature-extraction
    DebertaV2Model,feature-extraction
    DecisionTransformerModel,reinforcement-learning
    DeiTModel,image-feature-extraction
    DetrModel,image-feature-extraction
    Dinov2Model,image-feature-extraction
    DistilBertModel,feature-extraction
    DonutSwinModel,feature-extraction
    ElectraModel,feature-extraction
    EsmModel,feature-extraction
    GLPNModel,image-feature-extraction
    GPTBigCodeModel,feature-extraction
    GPTJModel,feature-extraction
    GPTNeoModel,feature-extraction
    GPTNeoXForCausalLM,text-generation
    GemmaForCausalLM,text-generation
    GraniteForCausalLM,text-generation
    GroupViTModel,feature-extraction
    HieraForImageClassification,image-classification
    HubertModel,feature-extraction
    IBertModel,feature-extraction
    ImageGPTModel,image-feature-extraction
    LayoutLMModel,feature-extraction
    LayoutLMv3Model,feature-extraction
    LevitModel,image-feature-extraction
    LiltModel,feature-extraction
    LlamaForCausalLM,text-generation
    LongT5Model,feature-extraction
    LongformerModel,feature-extraction
    MCTCTModel,feature-extraction
    MPNetModel,feature-extraction
    MT5Model,feature-extraction
    MarianMTModel,text2text-generation
    MarkupLMModel,feature-extraction
    MaskFormerForInstanceSegmentation,image-segmentation
    MegatronBertModel,feature-extraction
    MgpstrForSceneTextRecognition,feature-extraction
    MistralForCausalLM,text-generation
    MobileBertModel,feature-extraction
    MobileNetV1Model,image-feature-extraction
    MobileNetV2Model,image-feature-extraction
    MobileViTForImageClassification,image-classification
    ModernBertForMaskedLM,fill-mask
    MoonshineForConditionalGeneration,automatic-speech-recognition
    MptForCausalLM,text-generation
    MusicgenForConditionalGeneration,text-to-audio
    NystromformerModel,feature-extraction
    OPTModel,feature-extraction
    Olmo2ForCausalLM,text-generation
    OlmoForCausalLM,text-generation
    OwlViTModel,feature-extraction
    Owlv2Model,feature-extraction
    PatchTSMixerForPrediction,no-pipeline-tag
    PatchTSTForPrediction,no-pipeline-tag
    PegasusModel,feature-extraction
    Phi3ForCausalLM,text-generation
    PhiForCausalLM,text-generation
    Pix2StructForConditionalGeneration,image-to-text
    PoolFormerModel,image-feature-extraction
    PvtForImageClassification,image-classification
    Qwen2ForCausalLM,text-generation
    RTDetrForObjectDetection,object-detection
    RegNetModel,image-feature-extraction
    RemBertModel,feature-extraction
    ResNetForImageClassification,image-classification
    RoFormerModel,feature-extraction
    RobertaModel,feature-extraction
    RtDetrV2ForObjectDetection,object-detection
    SEWDModel,feature-extraction
    SEWModel,feature-extraction
    SamModel,mask-generation
    SegformerModel,image-feature-extraction
    SiglipModel,zero-shot-image-classification
    SiglipVisionModel,image-feature-extraction
    Speech2TextModel,feature-extraction
    SpeechT5ForTextToSpeech,text-to-audio
    SplinterModel,feature-extraction
    SqueezeBertModel,feature-extraction
    Swin2SRModel,image-feature-extraction
    SwinModel,image-feature-extraction
    Swinv2Model,image-feature-extraction
    T5ForConditionalGeneration,text2text-generation
    TableTransformerModel,image-feature-extraction
    UniSpeechForSequenceClassification,audio-classification
    ViTForImageClassification,image-classification
    ViTMAEModel,image-feature-extraction
    ViTMSNForImageClassification,image-classification
    VisionEncoderDecoderModel,document-question-answering
    VitPoseForPoseEstimation,keypoint-detection
    VitsModel,text-to-audio
    Wav2Vec2ConformerForCTC,automatic-speech-recognition
    Wav2Vec2Model,feature-extraction
    WhisperForConditionalGeneration,no-pipeline-tag
    XLMModel,feature-extraction
    XLMRobertaForCausalLM,text-generation
    YolosForObjectDetection,object-detection
    YolosModel,image-feature-extraction"""
)

__data_tasks__ = [
    "automatic-speech-recognition",
    "image-text-to-text",
    "image-to-text",
    "text-generation",
    "object-detection",
    "document-question-answering",
    "feature-extraction",
    "text-to-audio",
    "zero-shot-image-classification",
    "image-segmentation",
    "reinforcement-learning",
    "no-pipeline-tag",
    "image-classification",
    "text2text-generation",
    "mask-generation",
    "keypoint-detection",
    "audio-classification",
    "image-feature-extraction",
    "fill-mask",
]

__models_testing__ = """
hf-internal-testing/tiny-random-BeitForImageClassification
hf-internal-testing/tiny-random-convnext
fxmarty/tiny-random-GemmaForCausalLM
hf-internal-testing/tiny-random-GPTNeoXForCausalLM
hf-internal-testing/tiny-random-GraniteForCausalLM
hf-internal-testing/tiny-random-HieraForImageClassification
fxmarty/tiny-llama-fast-tokenizer
sshleifer/tiny-marian-en-de
hf-internal-testing/tiny-random-MaskFormerForInstanceSegmentation
echarlaix/tiny-random-mistral
hf-internal-testing/tiny-random-mobilevit
hf-internal-testing/tiny-random-MoonshineForConditionalGeneration
hf-internal-testing/tiny-random-OlmoForCausalLM
hf-internal-testing/tiny-random-Olmo2ForCausalLM
echarlaix/tiny-random-PhiForCausalLM
Xenova/tiny-random-Phi3ForCausalLM
fxmarty/pix2struct-tiny-random
fxmarty/tiny-dummy-qwen2
hf-internal-testing/tiny-random-ViTMSNForImageClassification
hf-internal-testing/tiny-random-YolosModel
hf-internal-testing/tiny-xlm-roberta
"""


@functools.cache
def load_models_testing() -> List[str]:
    """Returns model ids for testing."""
    return [_.strip() for _ in __models_testing__.split("\n") if _.strip()]


@functools.cache
def load_architecture_task() -> Dict[str, str]:
    """
    Returns a dictionary mapping architecture to task.

    import pprint
    from onnx_diagnostic.torch_models.hghub.hub_data import load_architecture_task
    pprint.pprint(load_architecture_task())
    """
    import pandas

    df = pandas.read_csv(io.StringIO(__data_arch__))
    return dict(zip(list(df["architecture"]), list(df["task"])))
