from transformers import AutoTokenizer, CLIPTextModel
import torch
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(
    [
        "a picture of an airport",
        "a picture of an american football field",
        "a picture of a baseball field",
        "a picture of a beach",
        "a picture of a bridge",
        "a picture of a cemetery",
        "a picture of a commercial area",
        "a picture of a dam",
        # "a picture of an equestrian facility",
        "a picture of a farmland or farm area",
        "a picture of a forest or nature reserve or dense trees",
        # "a picture of a garden",
        "a picture of a golf course",
        "a picture of a highway",
        "a picture of a marina",
        "a picture of a multi-story parking garage",
        "a picture of a park or recreational area",
        "a picture of a parking lot",
        "a picture of a pond or a lake",
        "a picture of a railroad",
        "a picture of a residential area or housing complex",
        "a picture of a river",
        "a picture of a roundabout intersection",
        "a picture of a sand area or sand bunker or sandy beach",
        "a picture of a shooting range",
        "a picture of a soccer field",
        "a picture of a supermarket",
        "a picture of a swimming pool or swimming hole",
        "a picture of a tennis court",
        "a picture of a university building",
        "a picture of a warehouse",
        "a picture of wetland",
        "a picture of a mansion",
        "a picture of a road",
    ],
    padding=True,
    return_tensors="pt",
)
# inputs = tokenizer(
#     [
#         "a picture of a golf course",
#         "a picture of a parking lot",
#         "a picture of a residential area or housing complex",
#     ],
#     padding=True,
#     return_tensors="pt",
# )
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled (EOS token) states
print(pooled_output)
print(pooled_output.shape)
torch.save(pooled_output, "mansion-all.pth")