import clip, torch

clipmodel, clippreprocess = clip.load("ViT-L/14@336px", device='cuda')

def get_clip_similarity_per_word(image, word_list):
    word_list = ["a photo of " + w for w in word_list]
    image = clippreprocess(image).unsqueeze(0).to('cuda')
    word_list = clip.tokenize(word_list).to("cuda")
    with torch.no_grad():
        image_features = clipmodel.encode_image(image)
        text_features = clipmodel.encode_text(word_list)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = image_features @ text_features.T
    return similarity.cpu().numpy().tolist()[0]
