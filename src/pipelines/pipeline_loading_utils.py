# coding=utf-8



import importlib
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from huggingface_hub import ModelCard, model_info
from huggingface_hub.utils import validate_hf_hub_args
from packaging import version

from .. import __version__
from ..utils import (
    FLAX_WEIGHTS_NAME,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    deprecate,
    get_class_from_dynamic_module,
    is_accelerate_available,
    is_peft_available,
    is_transformers_available,
    logging,
)
from ..utils.torch_utils import is_compiled_module


if is_transformers_available():
    import transformers
    from transformers import PreTrainedModel
    from transformers.utils import FLAX_WEIGHTS_NAME as TRANSFORMERS_FLAX_WEIGHTS_NAME
    from transformers.utils import SAFE_WEIGHTS_NAME as TRANSFORMERS_SAFE_WEIGHTS_NAME
    from transformers.utils import WEIGHTS_NAME as TRANSFORMERS_WEIGHTS_NAME

if is_accelerate_available():
    import accelerate
    from accelerate import dispatch_model
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import compute_module_sizes, get_max_memory


INDEX_FILE = "diffusion_pytorch_model.bin"
CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
DUMMY_MODULES_FOLDER = "diffusers.utils"
TRANSFORMERS_DUMMY_MODULES_FOLDER = "transformers.utils"
CONNECTED_PIPES_KEYS = ["prior"]

logger = logging.get_logger(__name__)

LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    "onnxruntime.training": {
        "ORTModule": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


def is_safetensors_compatible(filenames, passed_components=None, folder_names=None) -> bool:
    """
    Checking for safetensors compatibility:
    - The model is safetensors compatible only if there is a safetensors file for each model component present in
      filenames.

    Converting default pytorch serialized filenames to safetensors serialized filenames:
    - For models from the diffusers library, just replace the ".bin" extension with ".safetensors"
    - For models from the transformers library, the filename changes from "pytorch_model" to "model", and the ".bin"
      extension is replaced with ".safetensors"
    """
    passed_components = passed_components or []
    if folder_names is not None:
        filenames = {f for f in filenames if os.path.split(f)[0] in folder_names}

    # extract all components of the pipeline and their associated files
    components = {}
    for filename in filenames:
        if not len(filename.split("/")) == 2:
            continue

        component, component_filename = filename.split("/")
        if component in passed_components:
            continue

        components.setdefault(component, [])
        components[component].append(component_filename)

    # If there are no component folders check the main directory for safetensors files
    if not components:
        return any(".safetensors" in filename for filename in filenames)

    # iterate over all files of a component
    # check if safetensor files exist for that component
    # if variant is provided check if the variant of the safetensors exists
    for component, component_filenames in components.items():
        matches = []
        for component_filename in component_filenames:
            filename, extension = os.path.splitext(component_filename)

            match_exists = extension == ".safetensors"
            matches.append(match_exists)

        if not any(matches):
            return False

    return True


def variant_compatible_siblings(filenames, variant=None) -> Union[List[os.PathLike], str]:
    weight_names = [
        WEIGHTS_NAME,
        SAFETENSORS_WEIGHTS_NAME,
        FLAX_WEIGHTS_NAME,
        ONNX_WEIGHTS_NAME,
        ONNX_EXTERNAL_WEIGHTS_NAME,
    ]

    if is_transformers_available():
        weight_names += [TRANSFORMERS_WEIGHTS_NAME, TRANSFORMERS_SAFE_WEIGHTS_NAME, TRANSFORMERS_FLAX_WEIGHTS_NAME]

    # model_pytorch, diffusion_model_pytorch, ...
    weight_prefixes = [w.split(".")[0] for w in weight_names]
    # .bin, .safetensors, ...
    weight_suffixs = [w.split(".")[-1] for w in weight_names]
    # -00001-of-00002
    transformers_index_format = r"\d{5}-of-\d{5}"

    if variant is not None:
        # `diffusion_pytorch_model.fp16.bin` as well as `model.fp16-00001-of-00002.safetensors`
        variant_file_re = re.compile(
            rf"({'|'.join(weight_prefixes)})\.({variant}|{variant}-{transformers_index_format})\.({'|'.join(weight_suffixs)})$"
        )
        # `text_encoder/pytorch_model.bin.index.fp16.json`
        variant_index_re = re.compile(
            rf"({'|'.join(weight_prefixes)})\.({'|'.join(weight_suffixs)})\.index\.{variant}\.json$"
        )

    # `diffusion_pytorch_model.bin` as well as `model-00001-of-00002.safetensors`
    non_variant_file_re = re.compile(
        rf"({'|'.join(weight_prefixes)})(-{transformers_index_format})?\.({'|'.join(weight_suffixs)})$"
    )
    # `text_encoder/pytorch_model.bin.index.json`
    non_variant_index_re = re.compile(rf"({'|'.join(weight_prefixes)})\.({'|'.join(weight_suffixs)})\.index\.json")

    if variant is not None:
        variant_weights = {f for f in filenames if variant_file_re.match(f.split("/")[-1]) is not None}
        variant_indexes = {f for f in filenames if variant_index_re.match(f.split("/")[-1]) is not None}
        variant_filenames = variant_weights | variant_indexes
    else:
        variant_filenames = set()

    non_variant_weights = {f for f in filenames if non_variant_file_re.match(f.split("/")[-1]) is not None}
    non_variant_indexes = {f for f in filenames if non_variant_index_re.match(f.split("/")[-1]) is not None}
    non_variant_filenames = non_variant_weights | non_variant_indexes

    # all variant filenames will be used by default
    usable_filenames = set(variant_filenames)

    def convert_to_variant(filename):
        if "index" in filename:
            variant_filename = filename.replace("index", f"index.{variant}")
        elif re.compile(f"^(.*?){transformers_index_format}").match(filename) is not None:
            variant_filename = f"{filename.split('-')[0]}.{variant}-{'-'.join(filename.split('-')[1:])}"
        else:
            variant_filename = f"{filename.split('.')[0]}.{variant}.{filename.split('.')[1]}"
        return variant_filename

    def find_component(filename):
        if not len(filename.split("/")) == 2:
            return
        component = filename.split("/")[0]
        return component

    def has_sharded_variant(component, variant, variant_filenames):
        # If component exists check for sharded variant index filename
        # If component doesn't exist check main dir for sharded variant index filename
        component = component + "/" if component else ""
        variant_index_re = re.compile(
            rf"{component}({'|'.join(weight_prefixes)})\.({'|'.join(weight_suffixs)})\.index\.{variant}\.json$"
        )
        return any(f for f in variant_filenames if variant_index_re.match(f) is not None)

    for filename in non_variant_filenames:
        if convert_to_variant(filename) in variant_filenames:
            continue

        component = find_component(filename)
        # If a sharded variant exists skip adding to allowed patterns
        if has_sharded_variant(component, variant, variant_filenames):
            continue

        usable_filenames.add(filename)

    return usable_filenames, variant_filenames


@validate_hf_hub_args
def warn_deprecated_model_variant(pretrained_model_name_or_path, token, variant, revision, model_filenames):
    info = model_info(
        pretrained_model_name_or_path,
        token=token,
        revision=None,
    )
    filenames = {sibling.rfilename for sibling in info.siblings}
    comp_model_filenames, _ = variant_compatible_siblings(filenames, variant=revision)
    comp_model_filenames = [".".join(f.split(".")[:1] + f.split(".")[2:]) for f in comp_model_filenames]

    if set(model_filenames).issubset(set(comp_model_filenames)):
        warnings.warn(
            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` even though you can load it via `variant=`{revision}`. Loading model variants via `revision='{revision}'` is deprecated and will be removed in diffusers v1. Please use `variant='{revision}'` instead.",
            FutureWarning,
        )
    else:
        warnings.warn(
            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have the required variant filenames in the 'main' branch. \n The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {revision} files' so that the correct variant file can be added.",
            FutureWarning,
        )


def _unwrap_model(model):
    """Unwraps a model."""
    if is_compiled_module(model):
        model = model._orig_mod

    if is_peft_available():
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model = model.base_model.model

    return model


def maybe_raise_or_warn(
    library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
):
    """Simple helper method to raise or warn in case incorrect module has been passed"""
    if not is_pipeline_module:
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)
        class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

        expected_class_obj = None
        for class_name, class_candidate in class_candidates.items():
            if class_candidate is not None and issubclass(class_obj, class_candidate):
                expected_class_obj = class_candidate

        # Dynamo wraps the original model in a private class.
        # I didn't find a public API to get the original class.
        sub_model = passed_class_obj[name]
        unwrapped_sub_model = _unwrap_model(sub_model)
        model_cls = unwrapped_sub_model.__class__

        if not issubclass(model_cls, expected_class_obj):
            raise ValueError(
                f"{passed_class_obj[name]} is of type: {model_cls}, but should be" f" {expected_class_obj}"
            )
    else:
        logger.warning(
            f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
            " has the correct type"
        )


def get_class_obj_and_candidates(
    library_name, class_name, importable_classes, pipelines, is_pipeline_module, component_name=None, cache_dir=None
):
    """Simple helper method to retrieve class object of module as well as potential parent class objects"""
    component_folder = os.path.join(cache_dir, component_name)

    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)

        class_obj = getattr(pipeline_module, class_name)
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    elif os.path.isfile(os.path.join(component_folder, library_name + ".py")):
        # load custom component
        class_obj = get_class_from_dynamic_module(
            component_folder, module_file=library_name + ".py", class_name=class_name
        )
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    else:
        # else we just import it from the library.
        library = importlib.import_module(library_name)

        class_obj = getattr(library, class_name)
        class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

    return class_obj, class_candidates


def _get_custom_pipeline_class(
    custom_pipeline,
    repo_id=None,
    hub_revision=None,
    class_name=None,
    cache_dir=None,
    revision=None,
):
    if custom_pipeline.endswith(".py"):
        path = Path(custom_pipeline)
        # decompose into folder & file
        file_name = path.name
        custom_pipeline = path.parent.absolute()
    elif repo_id is not None:
        file_name = f"{custom_pipeline}.py"
        custom_pipeline = repo_id
    else:
        file_name = CUSTOM_PIPELINE_FILE_NAME

    if repo_id is not None and hub_revision is not None:
        # if we load the pipeline code from the Hub
        # make sure to overwrite the `revision`
        revision = hub_revision

    return get_class_from_dynamic_module(
        custom_pipeline,
        module_file=file_name,
        class_name=class_name,
        cache_dir=cache_dir,
        revision=revision,
    )


def _get_pipeline_class(
    class_obj,
    config=None,
    load_connected_pipeline=False,
    custom_pipeline=None,
    repo_id=None,
    hub_revision=None,
    class_name=None,
    cache_dir=None,
    revision=None,
):
    if custom_pipeline is not None:
        return _get_custom_pipeline_class(
            custom_pipeline,
            repo_id=repo_id,
            hub_revision=hub_revision,
            class_name=class_name,
            cache_dir=cache_dir,
            revision=revision,
        )

    if class_obj.__name__ != "DiffusionPipeline":
        return class_obj

    diffusers_module = importlib.import_module(class_obj.__module__.split(".")[0])
    class_name = class_name or config["_class_name"]
    if not class_name:
        raise ValueError(
            "The class name could not be found in the configuration file. Please make sure to pass the correct `class_name`."
        )

    class_name = class_name[4:] if class_name.startswith("Flax") else class_name

    pipeline_cls = getattr(diffusers_module, class_name)

    if load_connected_pipeline:
        from .auto_pipeline import _get_connected_pipeline

        connected_pipeline_cls = _get_connected_pipeline(pipeline_cls)
        if connected_pipeline_cls is not None:
            logger.info(
                f"Loading connected pipeline {connected_pipeline_cls.__name__} instead of {pipeline_cls.__name__} as specified via `load_connected_pipeline=True`"
            )
        else:
            logger.info(f"{pipeline_cls.__name__} has no connected pipeline class. Loading {pipeline_cls.__name__}.")

        pipeline_cls = connected_pipeline_cls or pipeline_cls

    return pipeline_cls


def _load_empty_model(
    library_name: str,
    class_name: str,
    importable_classes: List[Any],
    pipelines: Any,
    is_pipeline_module: bool,
    name: str,
    torch_dtype: Union[str, torch.dtype],
    cached_folder: Union[str, os.PathLike],
    **kwargs,
):
    # retrieve class objects.
    class_obj, _ = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        transformers_version = "N/A"

    # Determine library.
    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)

    model = None
    config_path = cached_folder
    user_agent = {
        "diffusers": __version__,
        "file_type": "model",
        "framework": "pytorch",
    }

    if is_diffusers_model:
        # Load config and then the model on meta.
        config, unused_kwargs, commit_hash = class_obj.load_config(
            os.path.join(config_path, name),
            cache_dir=cached_folder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            subfolder=kwargs.pop("subfolder", None),
            user_agent=user_agent,
        )
        with accelerate.init_empty_weights():
            model = class_obj.from_config(config, **unused_kwargs)
    elif is_transformers_model:
        config_class = getattr(class_obj, "config_class", None)
        if config_class is None:
            raise ValueError("`config_class` cannot be None. Please double-check the model.")

        config = config_class.from_pretrained(
            cached_folder,
            subfolder=name,
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            user_agent=user_agent,
        )
        with accelerate.init_empty_weights():
            model = class_obj(config)

    if model is not None:
        model = model.to(dtype=torch_dtype)
    return model


def _assign_components_to_devices(
    module_sizes: Dict[str, float], device_memory: Dict[str, float], device_mapping_strategy: str = "balanced"
):
    device_ids = list(device_memory.keys())
    device_cycle = device_ids + device_ids[::-1]
    device_memory = device_memory.copy()

    device_id_component_mapping = {}
    current_device_index = 0
    for component in module_sizes:
        device_id = device_cycle[current_device_index % len(device_cycle)]
        component_memory = module_sizes[component]
        curr_device_memory = device_memory[device_id]

        # If the GPU doesn't fit the current component offload to the CPU.
        if component_memory > curr_device_memory:
            device_id_component_mapping["cpu"] = [component]
        else:
            if device_id not in device_id_component_mapping:
                device_id_component_mapping[device_id] = [component]
            else:
                device_id_component_mapping[device_id].append(component)

            # Update the device memory.
            device_memory[device_id] -= component_memory
            current_device_index += 1

    return device_id_component_mapping


def _get_final_device_map(device_map, pipeline_class, passed_class_obj, init_dict, library, max_memory, **kwargs):
    # To avoid circular import problem.
    from diffusers import pipelines

    torch_dtype = kwargs.get("torch_dtype", torch.float32)

    # Load each module in the pipeline on a meta device so that we can derive the device map.
    init_empty_modules = {}
    for name, (library_name, class_name) in init_dict.items():
        if class_name.startswith("Flax"):
            raise ValueError("Flax pipelines are not supported with `device_map`.")

        # Define all importable classes
        is_pipeline_module = hasattr(pipelines, library_name)
        importable_classes = ALL_IMPORTABLE_CLASSES
        loaded_sub_model = None

        # Use passed sub model or load class_name from library_name
        if name in passed_class_obj:
            # if the model is in a pipeline module, then we load it from the pipeline
            # check that passed_class_obj has correct parent class
            maybe_raise_or_warn(
                library_name,
                library,
                class_name,
                importable_classes,
                passed_class_obj,
                name,
                is_pipeline_module,
            )
            with accelerate.init_empty_weights():
                loaded_sub_model = passed_class_obj[name]

        else:
            loaded_sub_model = _load_empty_model(
                library_name=library_name,
                class_name=class_name,
                importable_classes=importable_classes,
                pipelines=pipelines,
                is_pipeline_module=is_pipeline_module,
                pipeline_class=pipeline_class,
                name=name,
                torch_dtype=torch_dtype,
                cached_folder=kwargs.get("cached_folder", None),
                force_download=kwargs.get("force_download", None),
                proxies=kwargs.get("proxies", None),
                local_files_only=kwargs.get("local_files_only", None),
                token=kwargs.get("token", None),
                revision=kwargs.get("revision", None),
            )

        if loaded_sub_model is not None:
            init_empty_modules[name] = loaded_sub_model

    # determine device map
    # Obtain a sorted dictionary for mapping the model-level components
    # to their sizes.
    module_sizes = {
        module_name: compute_module_sizes(module, dtype=torch_dtype)[""]
        for module_name, module in init_empty_modules.items()
        if isinstance(module, torch.nn.Module)
    }
    module_sizes = dict(sorted(module_sizes.items(), key=lambda item: item[1], reverse=True))

    # Obtain maximum memory available per device (GPUs only).
    max_memory = get_max_memory(max_memory)
    max_memory = dict(sorted(max_memory.items(), key=lambda item: item[1], reverse=True))
    max_memory = {k: v for k, v in max_memory.items() if k != "cpu"}

    # Obtain a dictionary mapping the model-level components to the available
    # devices based on the maximum memory and the model sizes.
    final_device_map = None
    if len(max_memory) > 0:
        device_id_component_mapping = _assign_components_to_devices(
            module_sizes, max_memory, device_mapping_strategy=device_map
        )

        # Obtain the final device map, e.g., `{"unet": 0, "text_encoder": 1, "vae": 1, ...}`
        final_device_map = {}
        for device_id, components in device_id_component_mapping.items():
            for component in components:
                final_device_map[component] = device_id

    return final_device_map


def load_sub_model(
    library_name: str,
    class_name: str,
    importable_classes: List[Any],
    pipelines: Any,
    is_pipeline_module: bool,
    pipeline_class: Any,
    torch_dtype: torch.dtype,
    provider: Any,
    sess_options: Any,
    device_map: Optional[Union[Dict[str, torch.device], str]],
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]],
    offload_folder: Optional[Union[str, os.PathLike]],
    offload_state_dict: bool,
    model_variants: Dict[str, str],
    name: str,
    from_flax: bool,
    variant: str,
    low_cpu_mem_usage: bool,
    cached_folder: Union[str, os.PathLike],
    use_safetensors: bool,
):
    """Helper method to load the module `name` from `library_name` and `class_name`"""

    # retrieve class candidates

    class_obj, class_candidates = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    load_method_name = None
    # retrieve load method name
    for class_name, class_candidate in class_candidates.items():
        if class_candidate is not None and issubclass(class_obj, class_candidate):
            load_method_name = importable_classes[class_name][1]

    # if load method name is None, then we have a dummy module -> raise Error
    if load_method_name is None:
        none_module = class_obj.__module__
        is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER) or none_module.startswith(
            TRANSFORMERS_DUMMY_MODULES_FOLDER
        )
        if is_dummy_path and "dummy" in none_module:
            # call class_obj for nice error message of missing requirements
            class_obj()

        raise ValueError(
            f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
            f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
        )

    load_method = getattr(class_obj, load_method_name)

    # add kwargs to loading method
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    loading_kwargs = {}
    if issubclass(class_obj, torch.nn.Module):
        loading_kwargs["torch_dtype"] = torch_dtype
    if issubclass(class_obj, diffusers_module.OnnxRuntimeModel):
        loading_kwargs["provider"] = provider
        loading_kwargs["sess_options"] = sess_options

    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)

    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        transformers_version = "N/A"

    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )

    # When loading a transformers model, if the device_map is None, the weights will be initialized as opposed to diffusers.
    # To make default loading faster we set the `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
    # This makes sure that the weights won't be initialized which significantly speeds up loading.
    if is_diffusers_model or is_transformers_model:
        loading_kwargs["device_map"] = device_map
        loading_kwargs["max_memory"] = max_memory
        loading_kwargs["offload_folder"] = offload_folder
        loading_kwargs["offload_state_dict"] = offload_state_dict
        loading_kwargs["variant"] = model_variants.pop(name, None)
        loading_kwargs["use_safetensors"] = use_safetensors

        if from_flax:
            loading_kwargs["from_flax"] = True

        # the following can be deleted once the minimum required `transformers` version
        # is higher than 4.27
        if (
            is_transformers_model
            and loading_kwargs["variant"] is not None
            and transformers_version < version.parse("4.27.0")
        ):
            raise ImportError(
                f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
            )
        elif is_transformers_model and loading_kwargs["variant"] is None:
            loading_kwargs.pop("variant")

        # if `from_flax` and model is transformer model, can currently not load with `low_cpu_mem_usage`
        if not (from_flax and is_transformers_model):
            loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        else:
            loading_kwargs["low_cpu_mem_usage"] = False

    # check if the module is in a subdirectory
    if os.path.isdir(os.path.join(cached_folder, name)):
        loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
    else:
        # else load from the root directory
        loaded_sub_model = load_method(cached_folder, **loading_kwargs)

    if isinstance(loaded_sub_model, torch.nn.Module) and isinstance(device_map, dict):
        # remove hooks
        remove_hook_from_module(loaded_sub_model, recurse=True)
        needs_offloading_to_cpu = device_map[""] == "cpu"

        if needs_offloading_to_cpu:
            dispatch_model(
                loaded_sub_model,
                state_dict=loaded_sub_model.state_dict(),
                device_map=device_map,
                force_hooks=True,
                main_device=0,
            )
        else:
            dispatch_model(loaded_sub_model, device_map=device_map, force_hooks=True)

    return loaded_sub_model


def _fetch_class_library_tuple(module):
    # import it here to avoid circular import
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    pipelines = getattr(diffusers_module, "pipelines")

    # register the config from the original module, not the dynamo compiled one
    not_compiled_module = _unwrap_model(module)
    library = not_compiled_module.__module__.split(".")[0]

    # check if the module is a pipeline module
    module_path_items = not_compiled_module.__module__.split(".")
    pipeline_dir = module_path_items[-2] if len(module_path_items) > 2 else None

    path = not_compiled_module.__module__.split(".")
    is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

    # if library is not in LOADABLE_CLASSES, then it is a custom module.
    # Or if it's a pipeline module, then the module is inside the pipeline
    # folder so we set the library to module name.
    if is_pipeline_module:
        library = pipeline_dir
    elif library not in LOADABLE_CLASSES:
        library = not_compiled_module.__module__

    # retrieve class_name
    class_name = not_compiled_module.__class__.__name__

    return (library, class_name)


def _identify_model_variants(folder: str, variant: str, config: dict) -> dict:
    model_variants = {}
    if variant is not None:
        for sub_folder in os.listdir(folder):
            folder_path = os.path.join(folder, sub_folder)
            is_folder = os.path.isdir(folder_path) and sub_folder in config
            variant_exists = is_folder and any(p.split(".")[1].startswith(variant) for p in os.listdir(folder_path))
            if variant_exists:
                model_variants[sub_folder] = variant
    return model_variants


def _resolve_custom_pipeline_and_cls(folder, config, custom_pipeline):
    custom_class_name = None
    if os.path.isfile(os.path.join(folder, f"{custom_pipeline}.py")):
        custom_pipeline = os.path.join(folder, f"{custom_pipeline}.py")
    elif isinstance(config["_class_name"], (list, tuple)) and os.path.isfile(
        os.path.join(folder, f"{config['_class_name'][0]}.py")
    ):
        custom_pipeline = os.path.join(folder, f"{config['_class_name'][0]}.py")
        custom_class_name = config["_class_name"][1]

    return custom_pipeline, custom_class_name


def _maybe_raise_warning_for_inpainting(pipeline_class, pretrained_model_name_or_path: str, config: dict):
    if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
        version.parse(config["_diffusers_version"]).base_version
    ) <= version.parse("0.5.1"):
        from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy

        pipeline_class = StableDiffusionInpaintPipelineLegacy

        deprecation_message = (
            "You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the"
            f" {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For"
            " better inpainting results, we strongly suggest using Stable Diffusion's official inpainting"
            " checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting instead or adapting your"
            f" checkpoint {pretrained_model_name_or_path} to the format of"
            " https://huggingface.co/runwayml/stable-diffusion-inpainting. Note that we do not actively maintain"
            " the {StableDiffusionInpaintPipelineLegacy} class and will likely remove it in version 1.0.0."
        )
        deprecate("StableDiffusionInpaintPipelineLegacy", "1.0.0", deprecation_message, standard_warn=False)


def _update_init_kwargs_with_connected_pipeline(
    init_kwargs: dict, passed_pipe_kwargs: dict, passed_class_objs: dict, folder: str, **pipeline_loading_kwargs
) -> dict:
    from .pipeline_utils import DiffusionPipeline

    modelcard = ModelCard.load(os.path.join(folder, "README.md"))
    connected_pipes = {prefix: getattr(modelcard.data, prefix, [None])[0] for prefix in CONNECTED_PIPES_KEYS}

    # We don't scheduler argument to match the existing logic:
    # https://github.com/huggingface/diffusers/blob/867e0c919e1aa7ef8b03c8eb1460f4f875a683ae/src/diffusers/pipelines/pipeline_utils.py#L906C13-L925C14
    pipeline_loading_kwargs_cp = pipeline_loading_kwargs.copy()
    if pipeline_loading_kwargs_cp is not None and len(pipeline_loading_kwargs_cp) >= 1:
        for k in pipeline_loading_kwargs:
            if "scheduler" in k:
                _ = pipeline_loading_kwargs_cp.pop(k)

    def get_connected_passed_kwargs(prefix):
        connected_passed_class_obj = {
            k.replace(f"{prefix}_", ""): w for k, w in passed_class_objs.items() if k.split("_")[0] == prefix
        }
        connected_passed_pipe_kwargs = {
            k.replace(f"{prefix}_", ""): w for k, w in passed_pipe_kwargs.items() if k.split("_")[0] == prefix
        }

        connected_passed_kwargs = {**connected_passed_class_obj, **connected_passed_pipe_kwargs}
        return connected_passed_kwargs

    connected_pipes = {
        prefix: DiffusionPipeline.from_pretrained(
            repo_id, **pipeline_loading_kwargs_cp, **get_connected_passed_kwargs(prefix)
        )
        for prefix, repo_id in connected_pipes.items()
        if repo_id is not None
    }

    for prefix, connected_pipe in connected_pipes.items():
        # add connected pipes to `init_kwargs` with <prefix>_<component_name>, e.g. "prior_text_encoder"
        init_kwargs.update(
            {"_".join([prefix, name]): component for name, component in connected_pipe.components.items()}
        )

    return init_kwargs


def _get_custom_components_and_folders(
    pretrained_model_name: str,
    config_dict: Dict[str, Any],
    filenames: Optional[List[str]] = None,
    variant_filenames: Optional[List[str]] = None,
    variant: Optional[str] = None,
):
    config_dict = config_dict.copy()

    # retrieve all folder_names that contain relevant files
    folder_names = [k for k, v in config_dict.items() if isinstance(v, list) and k != "_class_name"]

    diffusers_module = importlib.import_module(__name__.split(".")[0])
    pipelines = getattr(diffusers_module, "pipelines")

    # optionally create a custom component <> custom file mapping
    custom_components = {}
    for component in folder_names:
        module_candidate = config_dict[component][0]

        if module_candidate is None or not isinstance(module_candidate, str):
            continue

        # We compute candidate file path on the Hub. Do not use `os.path.join`.
        candidate_file = f"{component}/{module_candidate}.py"

        if candidate_file in filenames:
            custom_components[component] = module_candidate
        elif module_candidate not in LOADABLE_CLASSES and not hasattr(pipelines, module_candidate):
            raise ValueError(
                f"{candidate_file} as defined in `model_index.json` does not exist in {pretrained_model_name} and is not a module in 'diffusers/pipelines'."
            )

    if len(variant_filenames) == 0 and variant is not None:
        error_message = f"You are trying to load the model files of the `variant={variant}`, but no such modeling files are available."
        raise ValueError(error_message)

    return custom_components, folder_names


def _get_ignore_patterns(
    passed_components,
    model_folder_names: List[str],
    model_filenames: List[str],
    variant_filenames: List[str],
    use_safetensors: bool,
    from_flax: bool,
    allow_pickle: bool,
    use_onnx: bool,
    is_onnx: bool,
    variant: Optional[str] = None,
) -> List[str]:
    if (
        use_safetensors
        and not allow_pickle
        and not is_safetensors_compatible(
            model_filenames, passed_components=passed_components, folder_names=model_folder_names
        )
    ):
        raise EnvironmentError(
            f"Could not find the necessary `safetensors` weights in {model_filenames} (variant={variant})"
        )

    if from_flax:
        ignore_patterns = ["*.bin", "*.safetensors", "*.onnx", "*.pb"]

    elif use_safetensors and is_safetensors_compatible(
        model_filenames, passed_components=passed_components, folder_names=model_folder_names
    ):
        ignore_patterns = ["*.bin", "*.msgpack"]

        use_onnx = use_onnx if use_onnx is not None else is_onnx
        if not use_onnx:
            ignore_patterns += ["*.onnx", "*.pb"]

        safetensors_variant_filenames = {f for f in variant_filenames if f.endswith(".safetensors")}
        safetensors_model_filenames = {f for f in model_filenames if f.endswith(".safetensors")}
        if len(safetensors_variant_filenames) > 0 and safetensors_model_filenames != safetensors_variant_filenames:
            logger.warning(
                f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n"
                f"[{', '.join(safetensors_variant_filenames)}]\nLoaded non-{variant} filenames:\n"
                f"[{', '.join(safetensors_model_filenames - safetensors_variant_filenames)}\nIf this behavior is not "
                f"expected, please check your folder structure."
            )

    else:
        ignore_patterns = ["*.safetensors", "*.msgpack"]

        use_onnx = use_onnx if use_onnx is not None else is_onnx
        if not use_onnx:
            ignore_patterns += ["*.onnx", "*.pb"]

        bin_variant_filenames = {f for f in variant_filenames if f.endswith(".bin")}
        bin_model_filenames = {f for f in model_filenames if f.endswith(".bin")}
        if len(bin_variant_filenames) > 0 and bin_model_filenames != bin_variant_filenames:
            logger.warning(
                f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n"
                f"[{', '.join(bin_variant_filenames)}]\nLoaded non-{variant} filenames:\n"
                f"[{', '.join(bin_model_filenames - bin_variant_filenames)}\nIf this behavior is not expected, please check "
                f"your folder structure."
            )

    return ignore_patterns
