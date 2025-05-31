import logging
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy.sd1_clip import escape_important, unescape_important, token_weights


te_config = {
    "l": {"start": 49406, "end": 49407, "pad": 0, 'max_tokens': 75},
    "g": {"start": 49406, "end": 49407, "pad": 0, 'max_tokens': 75},
}


def get_embeddings_tag(embeddings_dict: dict, weight = 1.0) -> dict:
    embeddings_dict_tag = {}
    for key in embeddings_dict.keys():
        if key not in te_config.keys():
            continue
        embeddings = embeddings_dict[key]
        embeddings_dict_tag[key] = embeddings_dict_tag.get(key, [])
        for embedding in embeddings:
            start = 0
            end = 0
            for i, (token, _) in enumerate(embedding):
                if token == te_config[key]["start"]:
                    start = i
                if (token == te_config[key]["pad"] or token == te_config[key]["end"]):
                    end = i
                    break
            if (end - start) < 2:
                continue
            embedding_tag = embedding[start + 1 : end]
            embedding_tag = [(token, weight) for token, _ in embedding_tag]
            embeddings_dict_tag[key].append(embedding_tag)
    
    return embeddings_dict_tag


class AlignedConditioning(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "text": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The text to be encoded.",
                    },
                ),
                "te": (
                    IO.CLIP,
                    {"tooltip": "The text encoder model used for encoding the text."},
                ),
                "sep": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "dynamicPrompts": False,
                        "tooltip": "The tag separator.",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = (
        "A tag aligned conditioning containing the embedded text used to guide the diffusion model.",
    )
    FUNCTION = "encode"

    CATEGORY = "custom_node_experiments"
    DESCRIPTION = "Encodes a text prompt in a tag aligned manner using a text encoder model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, te, text, sep):
        if te is None:
            raise RuntimeError(
                "ERROR: TE input is invalid: None\n\nIf the TE is from a checkpoint loader node your checkpoint does not contain a valid text encoder model."
            )
        escaped_text = escape_important(text)
        weighted_segments = token_weights(escaped_text, 1.0)
        weighted_tags = []
        for escaped_segment, weight in weighted_segments:
            segment = unescape_important(escaped_segment)
            segment = segment.strip().strip(sep)
            tags_segment = [tag.strip() for tag in segment.split(sep)]
            tags_segment_weighted = [(tag, weight) for tag in tags_segment if tag]
            weighted_tags.extend(tags_segment_weighted)
        
        embeddings_dict_default = te.tokenize(text)
        embeddings_dict_tags = {}
        for tag, weight in weighted_tags:
            embeddings_dict = te.tokenize(tag)
            for key in embeddings_dict.keys():
                embeddings_dict_tags[key] = embeddings_dict_tags.get(key, [])
            embeddings_dict_tag = get_embeddings_tag(embeddings_dict, weight)
            for key in embeddings_dict_tags:
                embeddings_dict_tags[key].extend(embeddings_dict_tag.get(key, []))
        
        embeddings_dict_sep = {}
        embeddings_dict = te.tokenize(sep)
        embeddings_dict_sep_full = get_embeddings_tag(embeddings_dict, 1.0)
        for key in embeddings_dict:
            embeddings_dict_sep[key] = (embeddings_dict_sep_full.get(key, [[]]))[0]

        embeddings_dict_final = {}
        for key in embeddings_dict_default.keys():
            if key not in te_config.keys():
                logging.warning(f'WARNING: Unsupported TE {key}, default embedding will be used')
                embeddings_dict_final[key] = embeddings_dict_default[key]
                continue
            
            embeddings_dict_final[key] = embeddings_dict_final.get(key, [])
            sep = embeddings_dict_sep[key]
            sep_len = len(sep)
            batch = []
            for embedding_tag in embeddings_dict_tags.get(key, []):
                if len(batch)==0:
                    if len(embedding_tag) <= te_config[key]['max_tokens']:
                        batch.extend(embedding_tag)
                    else:
                        logging.warning('WARNING: Tag embedding too long, ignoring')
                else:
                    if len(batch) + sep_len + len(embedding_tag) <= te_config[key]['max_tokens']:
                        batch.extend(sep)
                        batch.extend(embedding_tag)
                    else:
                        if len(batch)<te_config[key]['max_tokens']:
                            batch.extend([(te_config[key]['pad'], 1.0)]*(te_config[key]['max_tokens']-len(batch)))
                        batch_final = [(te_config[key]['start'], 1.0)] + batch + [(te_config[key]['end'], 1.0)]
                        embeddings_dict_final[key].append(batch_final)
                        batch = []
                        if len(embedding_tag) <= te_config[key]['max_tokens']:
                            batch.extend(embedding_tag)
                        else:
                            logging.warning('WARNING: Tag embedding too long, ignoring')

            if len(batch)>0:
                batch.extend([(te_config[key]['pad'], 1.0)]*(te_config[key]['max_tokens']-len(batch)))
                batch_final = [(te_config[key]['start'], 1.0)] + batch + [(te_config[key]['end'], 1.0)]
                embeddings_dict_final[key].append(batch_final)

        for key in embeddings_dict_final.keys():
            if len(embeddings_dict_final[key])==0:
                batch_final = [(te_config[key]['start'], 1.0)] + ([(te_config[key]['pad'], 1.0)]*te_config[key]['max_tokens']) + [(te_config[key]['end'], 1.0)]
                embeddings_dict_final[key].append(batch_final)

        return (te.encode_from_tokens_scheduled(embeddings_dict_final), )


NODE_CLASS_MAPPINGS = {
    "AlignedConditioning": AlignedConditioning,
}
